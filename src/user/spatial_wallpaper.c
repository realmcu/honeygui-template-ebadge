/*
 * Copyright (c) 2026, Realtek Semiconductor Corporation
 *
 * SPDX-License-Identifier: MIT
 */

/*============================================================================*
 *                          Spatial Wallpaper Demo
 *
 * 3D perspective-parallax wallpaper using ONLY a foreground + background image.
 *
 * Principle (mirrors the browser preview tool's 3D mode calculations):
 *   - Put the two images at different Z depths around a virtual "stage".
 *   - Each frame, rotate only the stage (rotateX / rotateY) by the device
 *     attitude. A pinhole-perspective projection then turns that single rotation
 *     into differential parallax (near layer moves more, far layer moves the
 *     opposite way) plus depth ordering.
 *   - A static reverse-scale s = (D - z)/D cancels the perspective magnify so
 *     the picture is pixel-aligned at rest.
 *   - Both layers are then grown by an `overscan` margin (cover-style) so that
 *     rotation/parallax never sweeps a layer edge into the frame -- otherwise
 *     the opaque background would expose a black border when tilted.
 *
 * Injection method:
 *   Each layer is a transform-node: a plain gui_obj container whose matrix we
 *   OVERWRITE with the per-layer screen-space homography inside its OBJ_PRE_PROCESS
 *   callback. obj_draw_prepare() runs a parent's prepare callback BEFORE its
 *   children inherit the parent matrix, so the zero-transform child image
 *   simply inherits the homography and draws with full perspective -- gui_img
 *   itself is left completely untouched.
 *
 * Attitude input -- 3-axis g-sensor (gravity components, int16):
 *   Widget frame : +x = left, +y = down, +z = out-of-screen (toward viewer).
 *   Sensor frame : +x = backward, +y = left, +z = up; gravity g points down.
 *   At rest (screen face-up) the frames align as
 *       x_ui(left) = y_acc ,  y_ui(down) = x_acc ,  z_ui(out) = z_acc ,
 *   so gravity expressed in widget axes is (gy, gx, gz). Its in-plane parts,
 *   normalized by |g|, give the left-right / up-down tilt that drives rotateY /
 *   rotateX. Normalizing by magnitude makes the result independent of the
 *   sensor's LSB-per-g scale and binds the zero pose to g = -z (screen up).
 *
 *   On real hardware, plug the sensor in one of two ways:
 *     - pull: register a reader with spatial_wallpaper_set_gsensor_reader();
 *             the attitude timer calls it every tick and feeds the result.
 *     - push: from a sampling loop or sensor ISR, call
 *             spatial_wallpaper_feed_gsensor() directly (and clear sim_enable).
 *   With no sensor on PC the demo synthesizes that data each tick so the exact
 *   same mapping path runs end-to-end. A registered reader overrides the sim.
 *============================================================================*/

/*============================================================================*
 *                        Header Files
 *============================================================================*/
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "gsensor_reader.h"
#include "guidef.h"
#include "gui_api.h"
#include "gui_fb.h"
#include "gui_img.h"
#include "gui_matrix.h"
#include "gui_obj.h"

/*============================================================================*
 *                           Resources
 *
 * Bins live in ./root_image and are linked in as objects by the SConscript
 * (add_prebuilt_binary_object). objcopy names the symbols after the file name
 * with every non-alphanumeric character replaced by '_'.
 *============================================================================*/

#define SW_BG_DATA ((const char *)"/background_clean.bin")
#define SW_FG_DATA ((const char *)"/foreground.bin")


/*============================================================================*
 *                            Macros
 *============================================================================*/
#define SW_DEG2RAD   (3.14159265358979f / 180.0f)

/*============================================================================*
 *                         Configuration Interface
 *
 * Group A maps 1:1 to the browser preview tool's tunables -- dial the effect
 * in the browser, then copy the numbers straight into the block below.
 *============================================================================*/
typedef struct
{
    /* ===== A. Browser-tunable parameters: copy values from the preview tool ===== */
    float persp;        /* D   camera distance            default 1200  */
    float tilt3d;       /* th  full tilt angle (degrees)  default 8     */
    float fg;           /*     foreground full travel px  default 50    */
    float bg;           /*     background full travel px  default 16    */
    float lerp;         /*     attitude smoothing coeff   default 0.12  */
    bool  invert;       /*     reverse parallax direction default false */

    /* ===== B. Hard-coded in the browser; rarely changed ===== */
    float fg_cap_ratio; /* fgZ clamp = ratio * D          default 0.6   */
    float overscan;     /* per-side edge margin (px). Browser 3D mode uses
                           max(max(bg, 2*blur), 44) + 8  ->  52 by default.
                           Larger values prevent exposed edges but zoom in more. */

    /* ===== C. Attitude input (3-axis g-sensor) ===== */
    bool     sim_enable;   /* PC demo: synthesize g-sensor data internally  */
    uint32_t timer_ms;     /* tick / redraw period (ms, ~60 fps)            */
    int16_t  g_one;        /* synthesizer only: raw counts per 1g           */
    float    sim_tilt_deg; /* synthesizer only: tilt wander amplitude (deg) */
    float    sim_speed;    /* synthesizer only: phase step per tick (rad)   */
    float    sim_freq_y;   /* synthesizer only: pitch/roll frequency ratio  */
    float    sim_phase_y;  /* synthesizer only: phase offset (rad)          */
} spatial_wallpaper_cfg_t;

/* Edit this block to paste a browser-tuned effect; rebuild to apply. */
static spatial_wallpaper_cfg_t g_cfg =
{
    .persp = 970.0f, .tilt3d = 8.0f, .fg = 50.0f, .bg = 16.0f,
    .lerp  = 0.12f,   .invert = false,
    .fg_cap_ratio = 0.6f, .overscan = 52.0f,
    .sim_enable = true, .timer_ms = 16, .g_one = 16384,
    .sim_tilt_deg = 45.0f, .sim_speed = 0.04f,
    .sim_freq_y = 0.73f, .sim_phase_y = 1.0f,
};
/*============================================================================*
 *                            Types & State
 *============================================================================*/
typedef struct
{
    gui_obj_t *stage;   /* transform-node whose matrix carries the homography */
    gui_img_t *img;     /* zero-transform child image                         */
    float      z;       /* precomputed signed depth (+ foreground, - background) */
} sw_layer_t;

typedef struct
{
    float phase;            /* synthesizer phase                          */
    float tgt_nx, tgt_ny;   /* target attitude from g-sensor, in [-1, 1]  */
    float cur_nx, cur_ny;   /* low-pass filtered attitude                 */
    float ax, ay;           /* rotateX / rotateY angle (degrees)          */
} sw_state_t;

/* Pull-style hook: a board callback returning one raw int16 gravity sample. */
typedef bool (*spatial_wallpaper_gsensor_reader_t)(int16_t *gx, int16_t *gy, int16_t *gz);

static sw_layer_t g_bg;
static sw_layer_t g_fg;
static sw_state_t g_state;
static gui_obj_t *g_ctrl;   /* invisible node that hosts the attitude timer */
static spatial_wallpaper_gsensor_reader_t g_reader;   /* live-hardware source, or NULL */

/*============================================================================*
 *                         Public API
 *
 * feed_gsensor() may be called from app code or a real sensor ISR; the demo's
 * own timer also calls it with synthesized data when sim_enable is set.
 *============================================================================*/
void spatial_wallpaper_feed_gsensor(int16_t gx, int16_t gy, int16_t gz);
void spatial_wallpaper_set_gsensor_reader(spatial_wallpaper_gsensor_reader_t reader);
void spatial_wallpaper_set_param(const spatial_wallpaper_cfg_t *cfg);

/*============================================================================*
 *                            Small Helpers
 *============================================================================*/
static float sw_clampf(float v, float lo, float hi)
{
    if (v < lo)
    {
        return lo;
    }
    if (v > hi)
    {
        return hi;
    }
    return v;
}

/*============================================================================*
 *                         Matrix Helpers
 *============================================================================*/

/**
 * @brief Build a 3D rotation matrix from tilt angles, using float trig.
 *
 * Same layout as matrix_compute_rotate() with rz = 0, i.e. R = Ry(ay)*Rx(ax),
 * but it uses cosf/sinf instead of the engine's integer-degree fix_sin/fix_cos
 * so that sub-degree motion within the +/-8 deg range stays perfectly smooth.
 */
static void sw_build_rotate(float ax_deg, float ay_deg, gui_matrix_t *r)
{
    float rx = ax_deg * SW_DEG2RAD;
    float ry = ay_deg * SW_DEG2RAD;
    float cx = cosf(rx), sx = sinf(rx);
    float cy = cosf(ry), sy = sinf(ry);

    r->m[0][0] = cy;    r->m[0][1] = sy * sx;   r->m[0][2] = sy * cx;
    r->m[1][0] = 0.0f;  r->m[1][1] = cx;        r->m[1][2] = -sx;
    r->m[2][0] = -sy;   r->m[2][1] = cy * sx;   r->m[2][2] = cy * cx;
}

/**
 * @brief Compute the image-local -> screen homography for one layer.
 *
 * Places a (w x h) layer, reverse-scaled and at depth z, centred on the stage;
 * rotates it about the stage centre by (ax, ay); then projects it through the
 * pinhole eye at (cx, cy, D) and solves the perspective transform. The result
 * maps image coords {0,0}..{w,h} directly to screen pixels.
 *
 * The reverse scale alone would project each layer to an exact (w x h) rect at
 * rest, leaving zero margin where image and screen share an edge (here: width).
 * So we additionally grow the layer by a cover-style factor `k` that guarantees
 * `over` px of margin on every side -- the homography equivalent of the
 * reference tool's "inset:-over + object-fit:cover". Both layers use the SAME
 * over (and have equal w/h), so their at-rest projections still coincide and
 * stay aligned; only the shared margin grows.
 */
static void sw_build_layer(float w, float h, float ax, float ay,
                           float z, float D, float cx, float cy,
                           float over, gui_matrix_t *out)
{
    gui_matrix_t R;
    sw_build_rotate(ax, ay, &R);

    float s = (D - z) / D;              /* reverse scale cancels perspective magnify */

    /* cover = max ratio that makes (w x h) span (screen + 2*over) on both axes;
     * screen w = 2*cx, screen h = 2*cy. Never shrink below the source image.   */
    float kw = 2.0f * (cx + over) / w;
    float kh = 2.0f * (cy + over) / h;
    float k  = (kw > kh) ? kw : kh;
    if (k < 1.0f)
    {
        k = 1.0f;
    }

    float hw = 0.5f * w * s * k;
    float hh = 0.5f * h * s * k;

    /* corners centred at origin, at depth z: TL, TR, BR, BL */
    gui_vertex_t v[4] =
    {
        { -hw, -hh, z },
        {  hw, -hh, z },
        {  hw,  hh, z },
        { -hw,  hh, z },
    };

    /* rotate about stage centre, then translate the centre to (cx, cy) */
    gui_vertex_t rv[4];
    for (int i = 0; i < 4; i++)
    {
        matrix_transfrom_rotate(&R, &v[i], &rv[i], cx, cy, 0.0f);
    }

    gui_vertex_t eye = { cx, cy, D };
    matrix_transfrom_blit(w, h, &eye, &rv[0], &rv[1], &rv[2], &rv[3], out);
}

/**
 * @brief Recompute static per-layer depths from the current config.
 *
 * Equivalent to the browser's update3dDepth(): Z = travel / tan(tilt), with the
 * foreground capped at fg_cap_ratio * D to avoid getting too close to the eye.
 */
static void sw_recompute_depth(void)
{
    float t = tanf(g_cfg.tilt3d * SW_DEG2RAD);
    if (t < 1e-3f)
    {
        t = 1e-3f;
    }

    float fg_z = g_cfg.fg / t;
    float cap  = g_cfg.fg_cap_ratio * g_cfg.persp;
    if (fg_z > cap)
    {
        fg_z = cap;
    }
    float bg_z = g_cfg.bg / t;

    g_fg.z = +fg_z;     /* foreground sits toward the eye   */
    g_bg.z = -bg_z;     /* background sits behind the screen */
}

/*============================================================================*
 *                         Per-Frame Injection
 *============================================================================*/

/* Build this layer's homography and overwrite the stage node's matrix with it. */
static void sw_layer_prepare(sw_layer_t *layer)
{
    if (layer == NULL || layer->img == NULL)
    {
        return;
    }

    float cx = gui_get_screen_width()  * 0.5f;
    float cy = gui_get_screen_height() * 0.5f;

    gui_matrix_t h;
    sw_build_layer((float)layer->img->base.w, (float)layer->img->base.h,
                   g_state.ax, g_state.ay, layer->z, g_cfg.persp, cx, cy,
                   g_cfg.overscan, &h);

    memcpy(layer->stage->matrix, &h, sizeof(gui_matrix_t));

    /*
     * Perspective can extend the image beyond the 360 x 360 stage, so clipping
     * must remain disabled to preserve the foreground edges.
     */
    layer->img->need_clip = false;
}

static void sw_stage_cb(gui_obj_t *obj, T_OBJ_CB_TYPE cb_type)
{
    if (cb_type == OBJ_PRE_PROCESS)
    {
        sw_layer_prepare((sw_layer_t *)obj->user_data);
    }
}

/* Synthesize a gentle tilt wander as int16 gravity (PC demo only). */
static void sw_simulate_gsensor(void)
{
    g_state.phase += g_cfg.sim_speed;

    /* two slow, out-of-phase tilt angles: roll about accel-x, pitch about accel-y */
    float a = g_cfg.sim_tilt_deg * SW_DEG2RAD * sinf(g_state.phase);
    float b = g_cfg.sim_tilt_deg * SW_DEG2RAD * sinf(g_cfg.sim_freq_y * g_state.phase
                                                     + g_cfg.sim_phase_y);

    /* gravity (0,0,-G) rotated into the sensor frame by Ry(b)*Rx(a):
     *   gx = -G*sin(b)*cos(a),  gy = G*sin(a),  gz = -G*cos(b)*cos(a)            */
    float G  = (float)g_cfg.g_one;
    float ca = cosf(a), sa = sinf(a);
    float cb = cosf(b), sb = sinf(b);

    int16_t gx = (int16_t)(-G * sb * ca);
    int16_t gy = (int16_t)(G * sa);
    int16_t gz = (int16_t)(-G * cb * ca);

    spatial_wallpaper_feed_gsensor(gx, gy, gz);
}

/* Timer callback: refresh attitude (sim or live), smooth it, map to rotation. */
static void sw_tick(void *obj)
{
    GUI_UNUSED(obj);

    if (g_reader != NULL)
    {
        /* live hardware: pull one raw sample and feed it */
        int16_t gx, gy, gz;
        if (g_reader(&gx, &gy, &gz))
        {
            spatial_wallpaper_feed_gsensor(gx, gy, gz);
        }
    }
    else if (g_cfg.sim_enable)
    {
        sw_simulate_gsensor();      /* no real sensor on PC: feed synthetic data */
    }
    /* else: attitude arrives via external spatial_wallpaper_feed_gsensor() push */

    /* one-pole low pass -- the same smoothing the browser applies to its input */
    g_state.cur_nx += (g_state.tgt_nx - g_state.cur_nx) * g_cfg.lerp;
    g_state.cur_ny += (g_state.tgt_ny - g_state.cur_ny) * g_cfg.lerp;

    /* rotateY's parallax runs opposite the roll direction under this engine's
     * projection, so left-right is negated to match the tilt; up-down already
     * agrees. invert flips both together for an overall direction reversal.   */
    float sign = g_cfg.invert ? 1.0f : -1.0f;
    g_state.ay = -sign * g_state.cur_nx * g_cfg.tilt3d;   /* rotateY <- left-right */
    g_state.ax = -sign * g_state.cur_ny * g_cfg.tilt3d;   /* rotateX <- up-down    */

    gui_fb_change();
}

/* Create one transform stage and its zero-transform image child. */
static void sw_create_layer(void *parent, sw_layer_t *layer, const char *stage_name,
                            const char *img_name, const unsigned char *img_data)
{
    gui_obj_t *root = parent;

    /* The stage matrix is replaced with the layer homography every frame. */
    layer->stage = gui_obj_create(root, stage_name, 0, 0, 0, 0);
    layer->stage->obj_cb = sw_stage_cb;
    layer->stage->has_pre_process_cb = true;
    layer->stage->user_data = layer;

    /* The child inherits the stage homography without an additional transform. */
    layer->img = gui_img_create_from_fs(layer->stage, img_name,
                                        (void *)img_data, 0, 0, 0, 0);
    layer->img->need_clip = false;
}

/*============================================================================*
 * Public Interface
 *============================================================================*/

/**
 * @brief Feed one raw 3-axis g-sensor sample (gravity components, int16).
 *
 * @param gx gravity along sensor +x (backward)
 * @param gy gravity along sensor +y (left)
 * @param gz gravity along sensor +z (up)
 *
 * Maps the sensor frame to the widget frame and extracts the normalized
 * left-right / up-down tilt that drives the parallax. Safe to call from an ISR.
 */
void spatial_wallpaper_feed_gsensor(int16_t gx, int16_t gy, int16_t gz)
{
    /*
     * Sensor to widget axes:
     * x_ui(left) = y_acc, y_ui(down) = x_acc, z_ui(out) = z_acc.
     */
    float ux = (float)gy;
    float uy = (float)gx;
    float uz = (float)gz;

    float magnitude = sqrtf(ux * ux + uy * uy + uz * uz);
    if (magnitude < 1.0f)
    {
        return;
    }

    /* Normalize tilt independently of the sensor's counts-per-g scale. */
    g_state.tgt_nx = sw_clampf(ux / magnitude, -1.0f, 1.0f);
    g_state.tgt_ny = sw_clampf(uy / magnitude, -1.0f, 1.0f);
}

/**
 * @brief Install a pull-style g-sensor reader.
 *
 * The attitude timer calls the reader every tick. Pass NULL to detach it and
 * use either the simulator or externally pushed samples.
 */
void spatial_wallpaper_set_gsensor_reader(spatial_wallpaper_gsensor_reader_t reader)
{
    g_reader = reader;
}

/**
 * @brief Apply a runtime configuration.
 *
 * Pass NULL to retain the current configuration and recompute derived depths.
 */
void spatial_wallpaper_set_param(const spatial_wallpaper_cfg_t *cfg)
{
    if (cfg != NULL)
    {
        g_cfg = *cfg;
    }
    sw_recompute_depth();
    gui_fb_change();
}

int spatial_wallpaper(gui_obj_t *parent)
{
#ifndef _HONEYGUI_SIMULATOR_
    spatial_wallpaper_set_gsensor_reader(gsensor_sc7a20_read_xyz);
#endif

    /* Compute static depths before drawing the first frame. */
    spatial_wallpaper_set_param(&g_cfg);

    /* Draw the background first, then alpha-blend the foreground over it. */
    sw_create_layer(parent, &g_bg, "sw_stage_bg", "sw_img_bg", SW_BG_DATA);
    sw_create_layer(parent, &g_fg, "sw_stage_fg", "sw_img_fg", SW_FG_DATA);

    /* One reload timer updates and smooths the current attitude. */
    g_ctrl = gui_obj_create(parent, "sw_ctrl", 0, 0, 1, 1);
    gui_obj_create_timer(g_ctrl, g_cfg.timer_ms, true, sw_tick);
    gui_obj_start_timer(g_ctrl);

    return 0;
}
