#include "bt_callbacks.h"
#include "../ui/bt_ui.h"
#include "../user/bt_user.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

// Timer animation counters
uint16_t circle_scale_timer_cnt = 0;
uint16_t bt_icon_timer_cnt = 0;
uint16_t view_transmit_timer_cnt = 0;

// Event callback function implementations

void bt_View_key_0_cb(void *obj, gui_event_t *e)
{
    GUI_UNUSED(obj);
    GUI_UNUSED(e);
    // Check key name
    if (strcmp(e->indev_name, "Menu") == 0)
    {
        gui_view_switch_direct(gui_view_get_current(), "easy_demoMainView", SWITCH_INIT_STATE, SWITCH_IN_NONE_ANIMATION);
    }
    else if (strcmp(e->indev_name, "Power") == 0)
    {
        gui_view_switch_direct(gui_view_get_current(), "easy_demoMainView", SWITCH_INIT_STATE, SWITCH_IN_NONE_ANIMATION);
    }
}

void qrcode_view_key_0_cb(void *obj, gui_event_t *e)
{
    GUI_UNUSED(obj);
    GUI_UNUSED(e);
    // Check key name
    if (strcmp(e->indev_name, "Menu") == 0)
    {
        gui_view_switch_direct(gui_view_get_current(), "easy_demoMainView", SWITCH_INIT_STATE, SWITCH_IN_NONE_ANIMATION);
    }
    else if (strcmp(e->indev_name, "Power") == 0)
    {
        gui_view_switch_direct(gui_view_get_current(), "easy_demoMainView", SWITCH_INIT_STATE, SWITCH_IN_NONE_ANIMATION);
    }
}

// Preset timer callback functions

/**
 * 动画 1
 * Component: circle_scale
 * Mode: Preset actions (multi-segment animation)
 * Segments: 2
 */
void circle_scale_timer_0_cb(void *obj)
{
    gui_obj_t *target = (gui_obj_t *)obj;
    const uint16_t total_cnt_max = 60;
    
    const uint16_t seg0_start = 0;
    const uint16_t seg0_end = 30;
    const uint16_t seg1_start = 30;
    const uint16_t seg1_end = 60;
    
    circle_scale_timer_cnt++;
    
    // Segment 1: 300ms, 1 action(s)
    if (circle_scale_timer_cnt > seg0_start && circle_scale_timer_cnt <= seg0_end) {
        uint16_t seg_cnt = circle_scale_timer_cnt - seg0_start;
        const uint16_t seg_cnt_max = seg0_end - seg0_start;
        
            // Adjust scale: (0.62, 0.62) -> (0.43, 0.43)
            const float zoom_x_origin = 0.62;
            const float zoom_x_target = 0.43;
            const float zoom_y_origin = 0.62;
            const float zoom_y_target = 0.43;
            float zoom_x_cur = zoom_x_origin + (zoom_x_target - zoom_x_origin) * seg_cnt / seg_cnt_max;
            float zoom_y_cur = zoom_y_origin + (zoom_y_target - zoom_y_origin) * seg_cnt / seg_cnt_max;
            gui_img_scale((gui_img_t *)target, zoom_x_cur, zoom_y_cur);
            
    }
    // Segment 2: 300ms, 1 action(s)
    else if (circle_scale_timer_cnt > seg1_start && circle_scale_timer_cnt <= seg1_end) {
        uint16_t seg_cnt = circle_scale_timer_cnt - seg1_start;
        const uint16_t seg_cnt_max = seg1_end - seg1_start;
        
            // Adjust scale: (0.43, 0.43) -> (0.62, 0.62)
            const float zoom_x_origin = 0.43;
            const float zoom_x_target = 0.62;
            const float zoom_y_origin = 0.43;
            const float zoom_y_target = 0.62;
            float zoom_x_cur = zoom_x_origin + (zoom_x_target - zoom_x_origin) * seg_cnt / seg_cnt_max;
            float zoom_y_cur = zoom_y_origin + (zoom_y_target - zoom_y_origin) * seg_cnt / seg_cnt_max;
            gui_img_scale((gui_img_t *)target, zoom_x_cur, zoom_y_cur);
            
    }
    
    if (circle_scale_timer_cnt >= total_cnt_max) {
        circle_scale_timer_cnt = 0; // Reset counter, continue loop
    }
}


/* @protected start custom_functions */
// Custom functions
void view_transmit_timer_0_cb(void *obj)
{
    GUI_UNUSED(obj);
    is_displaying_mainface = false;
}

static uint16_t bt_icon_idle_cnt = 0;
void bt_icon_timer_0_cb(void *obj)
{
    GUI_UNUSED(obj);
    
    is_displaying_mainface = false;

    void *src = NULL;
    if (is_bt_connect)
    {
        src = "/image/A8/bt_icon_conn.bin";
    }
    else
    {
        src = "/image/A8/bt_icon_discon.bin";
    }
    gui_img_set_src((gui_img_t *)bt_icon, src, IMG_SRC_FILESYS);

    bt_icon_idle_cnt++;
    if (bt_icon_idle_cnt >= 200)
    {
        bt_icon_idle_cnt = 0;
        if (is_bt_connect == false)
        {
            gui_view_switch_direct(gui_view_get_current(), "qrcode_view", SWITCH_OUT_NONE_ANIMATION, SWITCH_IN_NONE_ANIMATION);
            gui_obj_stop_timer((gui_obj_t *)bt_icon);
        }
    }
}
/* @protected end custom_functions */
