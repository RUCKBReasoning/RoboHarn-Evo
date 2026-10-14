window.DEMO_EVIDENCE = {
  "version": 6,
  "episodes": {
    "independent": {
      "1": {
        "decisions": [
          {
            "start": 0,
            "title": "Observe the new task.",
            "summary": "Read the instruction and inspect the scene before choosing an action.",
            "context": "Current instruction and camera observations.",
            "traceIndex": null,
            "taskLevel": "Identify the goal and next subtask.",
            "actionLevel": "Inspect the scene before choosing a motion."
          },
          {
            "start": 0.89741,
            "sourceSeconds": 134.6115596294403,
            "snappedAcrossCut": false,
            "traceIndex": 22,
            "context": "Three cups, three fixed assignments.",
            "title": "Cover red first.",
            "summary": "Use the initial-right cup. Release only when supported; withdraw and verify.",
            "taskLevel": "Cover red with its assigned cup.",
            "actionLevel": "Use the initial-right cup. Lower around red, support, release and withdraw."
          },
          {
            "start": 21.06148,
            "sourceSeconds": 694.0082235336304,
            "snappedAcrossCut": false,
            "traceIndex": 149,
            "context": "The red cover is complete.",
            "title": "Preserve the cup identities.",
            "summary": "Use the initial-middle cup for green; leave the completed red cover undisturbed.",
            "taskLevel": "Cover green while preserving the cup assignments.",
            "actionLevel": "Move the initial-middle cup over green. Support it before release, then verify enclosure."
          },
          {
            "start": 40.74966,
            "sourceSeconds": 1244.28959274292,
            "snappedAcrossCut": false,
            "traceIndex": 264,
            "context": "Red and green are covered.",
            "title": "Use the reserved cup for blue.",
            "summary": "Keep the initial-left cup binding; place, release and verify the final cover.",
            "taskLevel": "Cover blue with the reserved cup.",
            "actionLevel": "Move the initial-left cup over blue. Support, release and withdraw for verification."
          },
          {
            "start": 60.62309,
            "sourceSeconds": 1771.9878237247467,
            "snappedAcrossCut": false,
            "traceIndex": 384,
            "context": "The final global check is unresolved.",
            "title": "Look again before declaring success.",
            "summary": "Reobserve blue enclosure and all three cup assignments without disturbing them.",
            "taskLevel": "Verify all three covers before completion.",
            "actionLevel": "Reobserve blue enclosure and cup support. Leave every cover undisturbed."
          },
          {
            "start": 62.03734,
            "sourceSeconds": 1984.1259303092957,
            "snappedAcrossCut": false,
            "traceIndex": 461,
            "context": "The camera view is cropped.",
            "title": "Change the viewing angle.",
            "summary": "Adjust the empty arm, then reobserve the covers. Preserve every cup assignment.",
            "taskLevel": "Resolve the final cover check.",
            "actionLevel": "Adjust the empty arm to improve the view, then reobserve the covers."
          },
          {
            "start": 62.74218,
            "sourceSeconds": 2089.8519129753113,
            "snappedAcrossCut": false,
            "traceIndex": 513,
            "context": "The cropped view still limits verification.",
            "title": "Lift the empty gripper for a clearer view.",
            "summary": "Reobserve the final cover and reevaluate all three pairings.",
            "taskLevel": "Resolve the final cover check.",
            "actionLevel": "Lift the empty gripper for a clearer view, then reobserve all three pairings."
          }
        ],
        "adoptions": [],
        "timing": {
          "setting": "independent",
          "task": 1,
          "logged_span_sec": 2155.2912266254425,
          "first_execution_request_sec": 169.8428657054901,
          "completed_planner_turns": 6,
          "planner_turn_wall_sum_sec": 154.2617084980011,
          "execution_requests": 25,
          "active_execution_sec": 270.1068283290042,
          "first_tool": "move_ee_to_grounded_instance",
          "trace_last_event": "recovery_request"
        },
        "traceSha256": "bca07e59c9d6d3ce8fd95edfc9f8206a6fa4b7faaaa86870a8d98767a2ad6674"
      },
      "2": {
        "decisions": [
          {
            "start": 0,
            "title": "Observe the new task.",
            "summary": "Read the instruction and inspect the scene before choosing an action.",
            "context": "Current instruction and camera observations.",
            "traceIndex": null,
            "taskLevel": "Identify the goal and next subtask.",
            "actionLevel": "Inspect the scene before choosing a motion."
          },
          {
            "start": 1.34906,
            "sourceSeconds": 67.45317006111145,
            "snappedAcrossCut": false,
            "traceIndex": 22,
            "context": "The display reads 1 and 2.",
            "title": "Green: press 1 of 1.",
            "summary": "Approach with a closed gripper, press once, withdraw and await verification.",
            "taskLevel": "Green: enter press 1 of 1.",
            "actionLevel": "Close the gripper at clearance. Approach, press once, withdraw and verify."
          },
          {
            "start": 13.03577,
            "sourceSeconds": 262.1701982021332,
            "snappedAcrossCut": false,
            "traceIndex": 90,
            "context": "Green is verified at 1/1. Blue is 0/2.",
            "title": "Blue: press 1 of 2.",
            "summary": "Ground the blue target. Press once and verify before another press.",
            "taskLevel": "Blue: enter press 1 of 2.",
            "actionLevel": "Ground the blue target. Press once with the closed gripper, withdraw and verify."
          },
          {
            "start": 20.94764,
            "sourceSeconds": 434.4853398799896,
            "snappedAcrossCut": false,
            "traceIndex": 158,
            "context": "Blue is verified at 1/2.",
            "title": "Blue: press 2 of 2.",
            "summary": "Revalidate contact geometry, press once and await verification before red.",
            "taskLevel": "Blue: finish press 2 of 2 before red.",
            "actionLevel": "Revalidate contact geometry. Press once, withdraw and verify release."
          },
          {
            "start": 28.28936,
            "sourceSeconds": 600.2708339691162,
            "snappedAcrossCut": false,
            "traceIndex": 227,
            "context": "Green and blue counts are complete.",
            "title": "Red: confirm the counts.",
            "summary": "Press the red button once, withdraw and request the global check.",
            "taskLevel": "Red: confirm the completed counts.",
            "actionLevel": "Press red once with the closed gripper, then withdraw for the final check."
          },
          {
            "start": 39.21595,
            "sourceSeconds": 788.6345353126526,
            "snappedAcrossCut": false,
            "traceIndex": 293,
            "context": "Local press cycles are accepted.",
            "title": "Refresh the global check.",
            "summary": "Reobserve the numerals and buttons without repeating any press.",
            "taskLevel": "Verify the completed sequence.",
            "actionLevel": "Reobserve the numerals and buttons without repeating a press."
          }
        ],
        "adoptions": [],
        "timing": {
          "setting": "independent",
          "task": 2,
          "logged_span_sec": 803.19353556633,
          "first_execution_request_sec": 88.56166529655457,
          "completed_planner_turns": 5,
          "planner_turn_wall_sum_sec": 78.83974146842957,
          "execution_requests": 17,
          "active_execution_sec": 159.3639861799893,
          "first_tool": "close_gripper",
          "trace_last_event": "depth_source_decision"
        },
        "traceSha256": "04e62b57c18d7d5dae0a1bc330a654de56d97a30624fc5744b6bea3152a5e421"
      },
      "3": {
        "decisions": [
          {
            "start": 0,
            "title": "Observe the new task.",
            "summary": "Read the instruction and inspect the scene before choosing an action.",
            "context": "Current instruction and camera observations.",
            "traceIndex": null,
            "taskLevel": "Identify the goal and next subtask.",
            "actionLevel": "Inspect the scene before choosing a motion."
          },
          {
            "start": 2.26,
            "sourceSeconds": 108.60894989967346,
            "snappedAcrossCut": true,
            "traceIndex": 13,
            "context": "The blocks are hidden, so their counts are unknown.",
            "title": "Lift the cup before counting.",
            "summary": "Use the left arm to grasp and lift; keep the right arm stationary.",
            "taskLevel": "Uncover the blocks before counting.",
            "actionLevel": "Grasp and lift the cup with the left arm. Keep the right arm still."
          },
          {
            "start": 13.30268,
            "sourceSeconds": 601.010500907898,
            "snappedAcrossCut": true,
            "traceIndex": 91,
            "context": "The cup has been lifted.",
            "title": "Set it aside with support and clearance.",
            "summary": "Place away from blocks and buttons, release, then clear the other arm’s workspace.",
            "taskLevel": "Clear the workspace for button presses.",
            "actionLevel": "Set the cup away from blocks and buttons. Support, release and withdraw."
          },
          {
            "start": 20.62876,
            "sourceSeconds": 1054.4234437942505,
            "snappedAcrossCut": true,
            "traceIndex": 186,
            "context": "The observed counts are red 1, green 2, blue 1.",
            "title": "Red: press 1 of 1.",
            "summary": "Resolve the red button’s identity, press with the right arm and verify the result.",
            "taskLevel": "Red: enter press 1 of 1.",
            "actionLevel": "Bind the red button, press with the right arm and verify the result."
          },
          {
            "start": 30.84317,
            "sourceSeconds": 1316.747196674347,
            "snappedAcrossCut": true,
            "traceIndex": 268,
            "context": "Red is complete. Green needs two presses.",
            "title": "Green: press 1 of 2.",
            "summary": "Resolve the stale object label, press once and await verification.",
            "taskLevel": "Green: enter press 1 of 2.",
            "actionLevel": "Resolve the green target’s stale label. Press once, withdraw and verify."
          },
          {
            "start": 37.91315,
            "sourceSeconds": 1558.2858340740204,
            "snappedAcrossCut": true,
            "traceIndex": 351,
            "context": "One green press is verified.",
            "title": "Green: press 2 of 2.",
            "summary": "Revalidate the green target; verify a distinct press before advancing to blue.",
            "taskLevel": "Green: finish press 2 of 2 before blue.",
            "actionLevel": "Revalidate the green target, then verify a distinct press and withdrawal."
          },
          {
            "start": 45.39221,
            "sourceSeconds": 1857.7316300868988,
            "snappedAcrossCut": true,
            "traceIndex": 449,
            "context": "The green count is complete.",
            "title": "Blue: press 1 of 1.",
            "summary": "Resolve the blue target binding, press once and verify withdrawal.",
            "taskLevel": "Blue: enter press 1 of 1.",
            "actionLevel": "Resolve the blue target binding. Press once, release and verify."
          },
          {
            "start": 54.81221,
            "sourceSeconds": 2214.7626583576202,
            "snappedAcrossCut": true,
            "traceIndex": 549,
            "context": "The final blue actuation needs verification.",
            "title": "Review the existing press evidence.",
            "summary": "Keep both arms stationary; resolve the uncertainty before any additional contact.",
            "taskLevel": "Resolve the final blue press check.",
            "actionLevel": "Keep both arms still while reviewing the existing actuation evidence before further contact."
          }
        ],
        "adoptions": [],
        "timing": {
          "setting": "independent",
          "task": 3,
          "logged_span_sec": 2270.2837693691254,
          "first_execution_request_sec": 170.7108612060547,
          "completed_planner_turns": 7,
          "planner_turn_wall_sum_sec": 156.33245015144348,
          "execution_requests": 23,
          "active_execution_sec": 272.6360315750062,
          "first_tool": "move_ee_to_grounded_instance",
          "trace_last_event": "control_turn_start"
        },
        "traceSha256": "a9b4fc03c49de30032b984852e755a9b2807ea47a0e01dd1df6295219d4777b5"
      }
    },
    "rsi": {
      "1": {
        "decisions": [
          {
            "start": 0,
            "title": "Observe the new task.",
            "summary": "Read the instruction and inspect the scene before choosing an action.",
            "context": "Current instruction and camera observations.",
            "traceIndex": null,
            "taskLevel": "Identify the goal and next subtask.",
            "actionLevel": "Inspect the scene before choosing a motion."
          },
          {
            "start": 0.76125,
            "sourceSeconds": 114.18700981140137,
            "snappedAcrossCut": false,
            "traceIndex": 24,
            "context": "Red is the first target.",
            "title": "Cover red, then verify.",
            "summary": "Grasp a cup with verified geometry; place with table support, release and withdraw.",
            "taskLevel": "Cover red before green.",
            "actionLevel": "Grasp a cup, lower it around red, support it on the table, then release and withdraw."
          },
          {
            "start": 14.13159,
            "sourceSeconds": 598.7377502918243,
            "snappedAcrossCut": false,
            "traceIndex": 146,
            "context": "The red cover is complete.",
            "title": "Cover green next.",
            "summary": "Use the right arm; keep the completed red cover undisturbed.",
            "taskLevel": "Cover green before blue.",
            "actionLevel": "Use the right arm to place and release a supported cup. Verify the cover."
          },
          {
            "start": 28.11624,
            "sourceSeconds": 1121.4365754127502,
            "snappedAcrossCut": false,
            "traceIndex": 290,
            "context": "Red and green are covered.",
            "title": "Cover blue last.",
            "summary": "Use the left arm; verify the final cover while preserving the first two.",
            "taskLevel": "Cover blue while preserving the first two covers.",
            "actionLevel": "Use the left arm to place, support and release the cup. Withdraw and verify."
          }
        ],
        "adoptions": [],
        "timing": {
          "setting": "rsi",
          "task": 1,
          "logged_span_sec": 1511.6916899681091,
          "first_execution_request_sec": 138.18805623054504,
          "completed_planner_turns": 3,
          "planner_turn_wall_sum_sec": 57.988293409347534,
          "execution_requests": 24,
          "active_execution_sec": 275.8507909660002,
          "first_tool": "move_ee_to_grounded_instance",
          "trace_last_event": "terminal_observation_preprocess_skipped"
        },
        "traceSha256": "1daa799b8f07f421bde07034ab63e8ec6155eaedf08b5ec83baabc0e5d681538"
      },
      "2": {
        "decisions": [
          {
            "start": 0,
            "title": "Observe the new task.",
            "summary": "Read the instruction and inspect the scene before choosing an action.",
            "context": "Current instruction and camera observations.",
            "traceIndex": null,
            "taskLevel": "Identify the goal and next subtask.",
            "actionLevel": "Inspect the scene before choosing a motion."
          },
          {
            "start": 0.84117,
            "sourceSeconds": 126.1755599975586,
            "snappedAcrossCut": false,
            "traceIndex": 22,
            "context": "The display reads 2 and 1.",
            "title": "Green: press 1 of 2.",
            "summary": "Execute one depression and withdrawal; await verification before another press.",
            "taskLevel": "Green: enter press 1 of 2.",
            "actionLevel": "Close the gripper, approach from clearance, press once, withdraw and verify."
          },
          {
            "start": 7.81096,
            "sourceSeconds": 379.64331102371216,
            "snappedAcrossCut": false,
            "traceIndex": 99,
            "context": "One green press is verified.",
            "title": "Green: press 2 of 2.",
            "summary": "Complete the left count with a distinct press-and-release before selecting blue.",
            "taskLevel": "Green: finish press 2 of 2 before blue.",
            "actionLevel": "Use verified contact geometry for one distinct press-and-release, then await verification."
          },
          {
            "start": 14.17927,
            "sourceSeconds": 641.8911190032959,
            "snappedAcrossCut": false,
            "traceIndex": 179,
            "context": "Both green presses are verified.",
            "title": "Blue: enter the second count.",
            "summary": "Press once, withdraw and verify before the red confirmation.",
            "taskLevel": "Blue: enter press 1 of 1 before red.",
            "actionLevel": "Keep the gripper closed. Press once, withdraw to release and verify."
          },
          {
            "start": 20.39797,
            "sourceSeconds": 872.6959455013275,
            "snappedAcrossCut": false,
            "traceIndex": 255,
            "context": "Green 2/2 and blue 1/1 are verified.",
            "title": "Red: confirm the completed counts.",
            "summary": "Press red once and await verification; preserve the completed green and blue counts.",
            "taskLevel": "Red: confirm both completed counts.",
            "actionLevel": "Press red once, withdraw and await the final check."
          },
          {
            "start": 26.69016,
            "sourceSeconds": 1114.524477481842,
            "snappedAcrossCut": false,
            "traceIndex": 332,
            "context": "The red confirmation cycle is verified.",
            "title": "Refresh the global check.",
            "summary": "Reobserve the display and buttons without repeating any press.",
            "taskLevel": "Verify the completed sequence.",
            "actionLevel": "Reobserve the display and buttons. Preserve the counts without adding a press."
          },
          {
            "start": 27.67961,
            "sourceSeconds": 1262.9411420822144,
            "snappedAcrossCut": false,
            "traceIndex": 409,
            "context": "The global evaluation still needs review.",
            "title": "Inspect the task state again.",
            "summary": "Use a targeted view of both numbers and all buttons; preserve verified counts.",
            "taskLevel": "Resolve the global completion check.",
            "actionLevel": "Inspect both numerals and all buttons while preserving the arms and counts."
          }
        ],
        "adoptions": [
          {
            "start": 0.84117,
            "sourceSeconds": 126.17534470558167,
            "layer": "task",
            "atomicIndex": 6,
            "sourceLine": 10,
            "rule": "Finish the left count before the right.",
            "origin": "Press simulation → maintained store",
            "image": "assets/images/sim-press.jpg",
            "behaviorChanged": true
          },
          {
            "start": 1.24414,
            "sourceSeconds": 186.62127900123596,
            "layer": "action",
            "atomicIndex": 7,
            "sourceLine": 21,
            "rule": "Approach from above. Press, then release.",
            "origin": "Press simulation → maintained store",
            "image": "assets/images/sim-press.jpg",
            "behaviorChanged": false
          },
          {
            "start": 7.81095,
            "sourceSeconds": 379.64305686950684,
            "layer": "task",
            "atomicIndex": 6,
            "sourceLine": 50,
            "rule": "Finish the left count before the right.",
            "origin": "Press simulation → maintained store",
            "image": "assets/images/sim-press.jpg",
            "behaviorChanged": true
          },
          {
            "start": 8.26215,
            "sourceSeconds": 447.3228614330292,
            "layer": "action",
            "atomicIndex": 7,
            "sourceLine": 61,
            "rule": "Approach from above. Press, then release.",
            "origin": "Press simulation → maintained store",
            "image": "assets/images/sim-press.jpg",
            "behaviorChanged": false
          },
          {
            "start": 14.17927,
            "sourceSeconds": 641.8908500671387,
            "layer": "task",
            "atomicIndex": 7,
            "sourceLine": 91,
            "rule": "Enter the right count before confirmation.",
            "origin": "Press simulation → maintained store",
            "image": "assets/images/sim-press.jpg",
            "behaviorChanged": true
          },
          {
            "start": 14.6142,
            "sourceSeconds": 707.1303064823151,
            "layer": "action",
            "atomicIndex": 8,
            "sourceLine": 102,
            "rule": "Center the press. Withdraw to release.",
            "origin": "Press simulation → maintained store",
            "image": "assets/images/sim-press.jpg",
            "behaviorChanged": false
          },
          {
            "start": 20.39797,
            "sourceSeconds": 872.6956825256348,
            "layer": "task",
            "atomicIndex": 8,
            "sourceLine": 131,
            "rule": "Confirm only after both counts are complete.",
            "origin": "Press simulation → maintained store",
            "image": "assets/images/sim-press.jpg",
            "behaviorChanged": true
          },
          {
            "start": 20.86064,
            "sourceSeconds": 942.0967149734497,
            "layer": "action",
            "atomicIndex": 9,
            "sourceLine": 142,
            "rule": "One confirmation press, then withdraw.",
            "origin": "Press simulation → maintained store",
            "image": "assets/images/sim-press.jpg",
            "behaviorChanged": false
          }
        ],
        "timing": {
          "setting": "rsi",
          "task": 2,
          "logged_span_sec": 1319.0973234176636,
          "first_execution_request_sec": 130.83815145492554,
          "completed_planner_turns": 6,
          "planner_turn_wall_sum_sec": 197.71125173568726,
          "execution_requests": 17,
          "active_execution_sec": 161.8403576740002,
          "first_tool": "close_gripper",
          "trace_last_event": "perception_query_cache_invalidated"
        },
        "traceSha256": "3584ea24719c74ac6500996dd78f5d41450763ae9ca5506913b811425d841db1"
      },
      "3": {
        "decisions": [
          {
            "start": 0,
            "title": "Observe the new task.",
            "summary": "Read the instruction and inspect the scene before choosing an action.",
            "context": "Current instruction and camera observations.",
            "traceIndex": null,
            "taskLevel": "Identify the goal and next subtask.",
            "actionLevel": "Inspect the scene before choosing a motion."
          },
          {
            "start": 0.76607,
            "sourceSeconds": 114.91099548339844,
            "snappedAcrossCut": false,
            "traceIndex": 15,
            "context": "The blocks are hidden, so their counts are unknown.",
            "title": "Lift the cup before counting.",
            "summary": "Grasp with the left arm and lift along a validated path; keep the right arm still.",
            "taskLevel": "Uncover the blocks before counting.",
            "actionLevel": "Grasp with the left arm and lift along a validated path. Keep the right arm still."
          },
          {
            "start": 9.66911,
            "sourceSeconds": 694.3666369915009,
            "snappedAcrossCut": false,
            "traceIndex": 113,
            "context": "The exposed counts are red 2, green 1, blue 1.",
            "title": "Set the cup safely aside.",
            "summary": "Repair the placement bindings; release only with whole-cup support and clearance.",
            "taskLevel": "Clear the workspace for button presses.",
            "actionLevel": "Repair the placement target. Support the whole cup before release and withdrawal."
          },
          {
            "start": 18.17707,
            "sourceSeconds": 1340.5607886314392,
            "snappedAcrossCut": false,
            "traceIndex": 196,
            "context": "The cup is placed. Red needs two presses.",
            "title": "Prepare the red approach.",
            "summary": "Move the right arm to noncontact clearance; verify cup stability before pressing.",
            "taskLevel": "Prepare red press 1 of 2.",
            "actionLevel": "Move the right arm to noncontact clearance, then verify cup stability before pressing."
          },
          {
            "start": 22.92538,
            "sourceSeconds": 1593.8065662384033,
            "snappedAcrossCut": false,
            "traceIndex": 285,
            "context": "Red is the first count target.",
            "title": "Red: press 1 of 2.",
            "summary": "Use one distinct press-and-release. Await verification before the second press.",
            "taskLevel": "Red: enter press 1 of 2.",
            "actionLevel": "Use one distinct press-and-release. Await verification before repeating."
          },
          {
            "start": 28.34395,
            "sourceSeconds": 1875.5918414592743,
            "snappedAcrossCut": false,
            "traceIndex": 356,
            "context": "The first red press is verified.",
            "title": "Red: press 2 of 2.",
            "summary": "Verify a separate second depression and release before moving on to green.",
            "taskLevel": "Red: finish press 2 of 2 before green.",
            "actionLevel": "Execute a separate depression and release, then verify the result."
          },
          {
            "start": 33.4454,
            "sourceSeconds": 2136.8097565174103,
            "snappedAcrossCut": false,
            "traceIndex": 437,
            "context": "Red is complete. Green needs one press.",
            "title": "Bind the green target before contact.",
            "summary": "Resolve the stale label and contact geometry; press once and verify release.",
            "taskLevel": "Green: enter press 1 of 1 before blue.",
            "actionLevel": "Resolve the stale target label and contact geometry, then press once and verify release."
          },
          {
            "start": 36.46983,
            "sourceSeconds": 2410.4738471508026,
            "snappedAcrossCut": false,
            "traceIndex": 523,
            "context": "The green target binding is ambiguous.",
            "title": "Reobserve before descent.",
            "summary": "Refresh the green cap’s identity and geometry from the reached approach.",
            "taskLevel": "Keep green as the next count target.",
            "actionLevel": "Reobserve the green cap at the reached approach. Resolve its identity before descent."
          },
          {
            "start": 37.18758,
            "sourceSeconds": 2518.1367723941803,
            "snappedAcrossCut": false,
            "traceIndex": 564,
            "context": "The green cap has been re-grounded.",
            "title": "Retry the verified green target.",
            "summary": "Press once using refreshed geometry, then withdraw and await verification.",
            "taskLevel": "Complete the pending green press.",
            "actionLevel": "Use refreshed geometry to press once, withdraw and await verification."
          },
          {
            "start": 44.68723,
            "sourceSeconds": 2833.0840594768524,
            "snappedAcrossCut": false,
            "traceIndex": 640,
            "context": "Green is complete. Blue needs one press.",
            "title": "Blue: press 1 of 1.",
            "summary": "Confirm the individual blue binding, press once and verify release.",
            "taskLevel": "Blue: enter press 1 of 1.",
            "actionLevel": "Confirm the individual blue target. Press once, withdraw and verify release."
          },
          {
            "start": 52.30342,
            "sourceSeconds": 3336.5126206874847,
            "snappedAcrossCut": false,
            "traceIndex": 745,
            "context": "The blue actuation evidence is unresolved.",
            "title": "Review the existing blue cycle.",
            "summary": "Inspect the execution evidence and released cap without adding another press.",
            "taskLevel": "Resolve the final blue press check.",
            "actionLevel": "Review the existing contact evidence and released cap without adding a press."
          },
          {
            "start": 53.90531,
            "sourceSeconds": 3576.7964129447937,
            "snappedAcrossCut": false,
            "traceIndex": 817,
            "context": "A released cap cannot establish past actuation.",
            "title": "Request help resolving the evidence.",
            "summary": "Keep both arms stationary and preserve verified counts; do not infer an extra press.",
            "taskLevel": "Resolve the final blue press check.",
            "actionLevel": "Request help with missing actuation evidence. Keep both arms stationary."
          },
          {
            "start": 54.59571,
            "sourceSeconds": 3680.3568437099457,
            "snappedAcrossCut": false,
            "traceIndex": 862,
            "context": "Blue identity and historical actuation need review.",
            "title": "Refresh the blue observation.",
            "summary": "Submit fresh and recorded evidence to the verifier without another contact.",
            "taskLevel": "Resolve the final blue press check.",
            "actionLevel": "Submit fresh and recorded evidence to the verifier without another contact."
          }
        ],
        "adoptions": [
          {
            "start": 10.31147,
            "sourceSeconds": 787.3719806671143,
            "layer": "action",
            "atomicIndex": 6,
            "sourceLine": 82,
            "rule": "Support the cup before release. Keep the workspace clear.",
            "origin": "Cover simulation → maintained store",
            "image": "assets/images/sim-cover.jpg",
            "behaviorChanged": false
          },
          {
            "start": 22.92538,
            "sourceSeconds": 1593.8062810897827,
            "layer": "task",
            "atomicIndex": 8,
            "sourceLine": 160,
            "rule": "One press. Release. Verify before repeating.",
            "origin": "Maintained after Press by number",
            "image": "assets/images/task-2.jpg",
            "behaviorChanged": true
          },
          {
            "start": 23.52571,
            "sourceSeconds": 1683.8570811748505,
            "layer": "action",
            "atomicIndex": 7,
            "sourceLine": 171,
            "rule": "A centered press from above, followed by release.",
            "origin": "Maintained after Press by number",
            "image": "assets/images/task-2.jpg",
            "behaviorChanged": false
          },
          {
            "start": 28.34394,
            "sourceSeconds": 1875.5915524959564,
            "layer": "task",
            "atomicIndex": 9,
            "sourceLine": 201,
            "rule": "Verify the second press before switching targets.",
            "origin": "Maintained after Press by number",
            "image": "assets/images/task-2.jpg",
            "behaviorChanged": true
          },
          {
            "start": 28.74023,
            "sourceSeconds": 1935.0342161655426,
            "layer": "action",
            "atomicIndex": 7,
            "sourceLine": 212,
            "rule": "A centered press from above, followed by release.",
            "origin": "Maintained after Press by number",
            "image": "assets/images/task-2.jpg",
            "behaviorChanged": false
          },
          {
            "start": 33.88443,
            "sourceSeconds": 2202.6642365455627,
            "layer": "action",
            "atomicIndex": 7,
            "sourceLine": 252,
            "rule": "A centered press from above, followed by release.",
            "origin": "Maintained after Press by number",
            "image": "assets/images/task-2.jpg",
            "behaviorChanged": false
          },
          {
            "start": 37.74465,
            "sourceSeconds": 2601.697462797165,
            "layer": "action",
            "atomicIndex": 7,
            "sourceLine": 313,
            "rule": "A centered press from above, followed by release.",
            "origin": "Maintained after Press by number",
            "image": "assets/images/task-2.jpg",
            "behaviorChanged": false
          },
          {
            "start": 45.17697,
            "sourceSeconds": 2906.5447909832,
            "layer": "action",
            "atomicIndex": 7,
            "sourceLine": 353,
            "rule": "A centered press from above, followed by release.",
            "origin": "Maintained after Press by number",
            "image": "assets/images/task-2.jpg",
            "behaviorChanged": false
          }
        ],
        "timing": {
          "setting": "rsi",
          "task": 3,
          "logged_span_sec": 3636.8607347011566,
          "first_execution_request_sec": 154.07482361793518,
          "completed_planner_turns": 12,
          "planner_turn_wall_sum_sec": 537.9457812309265,
          "execution_requests": 24,
          "active_execution_sec": 250.353674253005,
          "first_tool": "move_ee_to_grounded_instance",
          "trace_last_event": "depth_source_decision"
        },
        "traceSha256": "4b2b13de01971ee159e60ad0fd969bbc70b16449e307dc1261dfae1f542dce33"
      }
    }
  },
  "tasks": {
    "1": {
      "defaultMoment": "red",
      "moments": [
        {
          "id": "start",
          "label": "From the start",
          "independent": 0,
          "rsi": 0
        },
        {
          "id": "red",
          "label": "Cover red",
          "independent": 0.9774,
          "rsi": 0.8412
        },
        {
          "id": "green",
          "label": "Cover green",
          "independent": 21.1415,
          "rsi": 14.2116
        },
        {
          "id": "blue",
          "label": "Cover blue",
          "independent": 40.8297,
          "rsi": 28.1962
        }
      ]
    },
    "2": {
      "defaultMoment": "first",
      "moments": [
        {
          "id": "start",
          "label": "From the start",
          "independent": 0,
          "rsi": 0
        },
        {
          "id": "first",
          "label": "First counted press",
          "independent": 1.4291,
          "rsi": 0.9212
        },
        {
          "id": "complete",
          "label": "Complete a count",
          "independent": 21.0276,
          "rsi": 7.891
        },
        {
          "id": "confirm",
          "label": "Confirm the counts",
          "independent": 28.3694,
          "rsi": 20.478
        }
      ]
    },
    "3": {
      "defaultMoment": "first",
      "moments": [
        {
          "id": "start",
          "label": "From the start",
          "independent": 0,
          "rsi": 0
        },
        {
          "id": "place",
          "label": "Set the cup aside",
          "independent": 13.3827,
          "rsi": 10.3915
        },
        {
          "id": "first",
          "label": "First counted press",
          "independent": 20.7088,
          "rsi": 23.0054
        },
        {
          "id": "complete",
          "label": "Complete a count",
          "independent": 37.9931,
          "rsi": 28.4239
        }
      ]
    }
  },
  "definitions": {
    "first_execution_request_sec": "From instruction_set to first logged request_started_at in execution_results. Both Press runs begin with close_gripper; Cover and Uncover begin with move_ee_to_grounded_instance. Dispatch latency, not first physical motion.",
    "logged_span_sec": "From instruction_set to last saved trace event. Includes interrupted runs; not successful completion time.",
    "completed_planner_turns": "Number of control_turn_result events. Does not measure CoT tokens.",
    "planner_turn_wall_sum_sec": "Sum of matched control_turn_start to control_turn_result intervals; includes retrieval, service and network time. Not isolated reasoning generation time."
  },
  "scope": "Current non-matched demonstrations do not establish shorter CoT or generally lower latency with RSI. First execution request latency is mixed. Adopted knowledge and its provenance can be demonstrated directly."
};
