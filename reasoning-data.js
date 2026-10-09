window.REASONING_DATA = {
  "1": [
    {
      "start": 0,
      "end": 27.667,
      "title": "Grasp the first cup",
      "plan": "Cover red first. Keep the other arm clear.",
      "action": "Grasp the first cup with the left arm.",
      "check": "Establish a grasp before transporting the cup."
    },
    {
      "start": 27.667,
      "end": 56.333,
      "title": "Cover red",
      "plan": "Complete the red cover before moving to green.",
      "action": "Lower the cup opening-down over the red block, then release.",
      "check": "Check table support, release and coverage."
    },
    {
      "start": 56.333,
      "end": 77.667,
      "title": "Grasp with the other arm",
      "plan": "Use the other arm for the next cover.",
      "action": "Grasp the next cup with the right arm.",
      "check": "Keep the completed red cover undisturbed."
    },
    {
      "start": 77.667,
      "end": 86.0,
      "title": "Lift clear",
      "plan": "Clear the tabletop before transporting the cup.",
      "action": "Lift the held cup along a clear path.",
      "check": "Preserve the grasp and avoid nearby objects."
    },
    {
      "start": 86.0,
      "end": 114.667,
      "title": "Cover green",
      "plan": "Cover green before starting the final blue cover.",
      "action": "Lower the cup over green, release and withdraw.",
      "check": "Check that the green block remains covered."
    },
    {
      "start": 114.667,
      "end": 149.333,
      "title": "Grasp the final cup",
      "plan": "Finish the red–green–blue sequence.",
      "action": "Grasp the remaining cup.",
      "check": "Keep both completed covers stable."
    },
    {
      "start": 149.333,
      "end": 184.0,
      "title": "Cover blue",
      "plan": "Complete the last cover.",
      "action": "Place the final cup over the blue block and release.",
      "check": "Check all three covers and the required order."
    }
  ],
  "2": [
    {
      "start": 0,
      "end": 29.333,
      "title": "Green · press 1 of 2",
      "plan": "The displayed counts are green ×2 and blue ×1. Start with green.",
      "action": "Close the empty gripper, press green once and withdraw.",
      "check": "Count one distinct press-and-release cycle."
    },
    {
      "start": 29.333,
      "end": 55.0,
      "title": "Green · press 2 of 2",
      "plan": "One green press remains before switching targets.",
      "action": "Press green once more, then withdraw clear.",
      "check": "Check the second cycle completes green ×2."
    },
    {
      "start": 55.0,
      "end": 81.0,
      "title": "Blue · press 1 of 1",
      "plan": "The green count is complete. Enter blue ×1.",
      "action": "Approach blue, press once and release.",
      "check": "Check the blue count before confirmation."
    },
    {
      "start": 81.0,
      "end": 107.0,
      "title": "Red · confirm",
      "plan": "Confirm after both displayed counts are entered.",
      "action": "Press the red confirmation button once and withdraw.",
      "check": "Check green ×2 → blue ×1 → red confirmation."
    }
  ],
  "3": [
    {
      "start": 0,
      "end": 21.0,
      "title": "Grasp the cup",
      "plan": "Reveal the blocks before counting. Keep the right arm stationary.",
      "action": "Grasp the covering cup with the left arm.",
      "check": "Establish a grasp before lifting."
    },
    {
      "start": 21.0,
      "end": 28.0,
      "title": "Uncover the blocks",
      "plan": "Uncover the full set of blocks.",
      "action": "Lift the cup clear while preserving the grasp.",
      "check": "Observe the exposed blocks before planning the presses."
    },
    {
      "start": 28.0,
      "end": 44.0,
      "title": "Place the cup",
      "plan": "Set the cup aside without covering blocks or obstructing buttons.",
      "action": "Approach a supported, clear placement from above and release.",
      "check": "Check cup support, release and workspace clearance."
    },
    {
      "start": 44.0,
      "end": 51.333,
      "title": "Withdraw the first arm",
      "plan": "Clear the first arm before handing over to the other arm.",
      "action": "Withdraw the open left gripper from the button workspace.",
      "check": "Keep the cup stable and all blocks exposed."
    },
    {
      "start": 51.333,
      "end": 68.333,
      "title": "Prepare the other arm",
      "plan": "Retain the block counts: red ×2, green ×1, blue ×1.",
      "action": "Close the empty right gripper and approach red.",
      "check": "Keep the left arm clear; prepare contact without pressing yet."
    },
    {
      "start": 68.333,
      "end": 88.0,
      "title": "Red · press 1 of 2",
      "plan": "Complete two distinct red cycles before moving to green.",
      "action": "Press red once, release and withdraw.",
      "check": "Verify the first cycle before the second press."
    },
    {
      "start": 88.0,
      "end": 106.667,
      "title": "Red · press 2 of 2",
      "plan": "One red cycle remains. Preserve the completed first press.",
      "action": "Press red a second time, then release and withdraw.",
      "check": "Verify the second cycle before advancing to green."
    },
    {
      "start": 106.667,
      "end": 113.333,
      "title": "Green · approach",
      "plan": "Red is complete. Identify the green button in the current view.",
      "action": "Move to a noncontact approach above green.",
      "check": "Refresh the button identity and contact geometry before descent."
    },
    {
      "start": 113.333,
      "end": 143.333,
      "title": "Green · rebind and press",
      "plan": "Use the refreshed green-button geometry. Preserve the red count.",
      "action": "Press green once and withdraw with the gripper closed.",
      "check": "Check the press-and-release cycle before moving to blue."
    },
    {
      "start": 143.333,
      "end": 159.0,
      "title": "Blue · press and release",
      "plan": "Finish with blue ×1, preserving the completed red and green counts.",
      "action": "Approach the blue cap with the closed gripper and press.",
      "check": "Check contact and release without repeating completed presses."
    },
    {
      "start": 159.0,
      "end": 167.0,
      "title": "Blue · withdraw",
      "plan": "Complete the composed task with the cup and blocks undisturbed.",
      "action": "Withdraw the right gripper to leave the buttons clear.",
      "check": "Check cup placement, exposed blocks and the required press sequence."
    }
  ]
};
