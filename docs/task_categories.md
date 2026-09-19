# ActiveArena Task Categories

ActiveArena groups its tasks into five categories according to the number of
objects involved, the structure of the perception--action loop, and whether
the robot must interact with the scene to reveal information.

## Category definitions

1. **Single-Object Search (SS).** Locate a single target and immediately
   perform the required manipulation. These tasks require the robot to use its
   limited memory to identify unexplored regions.
2. **Single-Object Loop (SL).** Complete cross-region search and the transport
   of a single object.
3. **Multi-Object Decision (MD).** Search among multiple candidates and select
   the correct target or action from the acquired evidence.
4. **Multi-Object Loop (ML).** Complete multiple perception--action loops
   across objects and spatial regions.
5. **Interactive Information Acquisition (IA).** Physically interact with the
   environment to reveal otherwise hidden task-relevant information before
   determining the appropriate manipulation.

Representative task rollouts are shown in the benchmark figure in the project
README.

## Task lists

### SS — Single-Object Search

1. `click_bell_rotate_view`
2. `press_stapler_rotate_view`
3. `shake_bottle_horizontally_rotate_view`
4. `shake_bottle_rotate_view`
5. `turn_switch_rotate_view`

### SL — Single-Object Loop

1. `beat_block_hammer_rotate_view`
2. `place_container_plate_rotate_view`
3. `place_empty_cup_rotate_view`
4. `place_fan_rotate_view`
5. `place_mouse_pad_rotate_view`
6. `place_object_basket_fan_double`
7. `place_object_scale_rotate_view`
8. `place_object_stand_rotate_view`
9. `place_shoe_rotate_view`
10. `put_block_on_upper_easy`
11. `move_pillbottle_pad_rotate_view`
12. `move_stapler_pad_rotate_view`
13. `stack_blocks_two_rotate_view`
14. `place_a2b_left_rotate_view`
15. `stamp_seal_rotate_view`
16. `place_a2b_right_rotate_view`

### ML — Multi-Object Loop

1. `blocks_ranking_rgb_fan_double`
2. `blocks_ranking_rgb_rotate_view`
3. `blocks_ranking_size_fan_double`
4. `blocks_ranking_size_rotate_view`
5. `put_block_on_upper_hard`
6. `place_cans_plasticbox_rotate_view`

### IA — Interactive Information Acquisition

1. `check_block_color`
2. `check_cola_color`
3. `check_cola_date`
4. `rank_backside_rgb_blocks`
5. `match_backside_two_blocks`

### MD — Multi-Object Decision

1. `count_color_kinds_press_button`
2. `count_random_object_press_button`
3. `count_target_press_button`
