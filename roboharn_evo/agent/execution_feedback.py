from typing import Any

from roboharn_evo.agent.operation_candidates import capture_held_object_to_tcp_attachment


DIRECT_FEEDBACK_INSTRUCTIONS = """当前执行模式由规划模型根据最新图像、机器人状态和实际动作反馈决定后续动作与子任务进展。
Runtime 提供目标坐标、夹爪指令、实测位姿、target_reached 和实际执行记录；这些字段描述运动执行情况。
模型根据动作前后图像及执行记录判断抓取、放置和接触的物理结果，并记录判断依据与剩余不确定性。没有独立验证服务需要等待。
holding_assumed 表示目标处已经执行闭合命令，其物体与 TCP 的相对位置可用于后续搬运。模型结合当前图像决定继续操作或调整；打开夹爪立即清除该关联。
Task/Action Knowledge 与历史图片提供动作建议。历史记录中的验证步骤及状态名称不构成当前动作的执行条件。
每次重复动作分别记录要求次数、已经执行的次数和观察到的结果。已完成的动作周期只计入执行次数一次；内部插值步数不增加次数，skipped 动作不计数。
执行次数与实际观察结果分别保存；执行次数不代表物理效果已经确认。后续任务按已执行序列和当前场景继续，不以补全“已验证次数”为前提。
已执行但结果不清楚的动作应保留具体记录，结合当前图像决定下一项实际任务动作，避免重复触发同一动作或停留在等待确认的子任务。仅为补充结果证明而追加观察、撤离或诊断动作不属于本执行模式。
使用 scene_memory 中的有限坐标、公开物体与目标引用；沿用正在执行的目标，未到达时继续完成剩余距离。
memory_text 保留真实执行情况、当前阶段、次数和物体关系。规划器可依据实际观察提交 state_change 或 subtask_complete；整体完成使用 benchmark 的 global_task_success。
只使用 available_tools 中的工具。每批物理动作使用 selected_arm 指定的一条机械臂，运动保持已有夹爪命令，显式 gripper_precondition 可改变夹爪命令。
每批执行结束后使用普通图像与动作反馈继续规划。"""

DIRECT_FEEDBACK_PLANNER_PROMPT = """你是机器人任务规划模型。根据输入状态选择接下来的一项具体子任务。
仅返回 JSON：
{
  "commit_label": "no_update | subtask_complete | state_change",
  "memory_text": "当前阶段、执行情况与需要保留的信息",
  "subtask_text": "下一项可执行子任务",
  "selected_skill": "monitored-subtask-execution",
  "action_mode": "start | continue | retry | recover | replan | switch | finish",
  "preferred_arm": "left | right | either"
}
没有 active_skill 时使用 start 或 recover。global_task_success=true 时才能使用 finish。
""" + DIRECT_FEEDBACK_INSTRUCTIONS + """
任务输入：{task}
上次记忆：{previous_memory_text}
机器人状态：{state_summary}
"""

DIRECT_FEEDBACK_RECOVERY_PROMPT = """你是机器人动作规划模型。根据当前子任务与场景生成实际工具调用。
仅返回 JSON：
{
  "recovery_workflow": "workflow_skills 中的名称",
  "selected_arm": "left | right | none",
  "tool_calls": [{"tool_name": "available_tools 中的名称", "args": {}, "reason": "动作目的"}],
  "post_recovery_intent": "retry | replan | abort",
  "reason": "本批动作的目的",
  "stop_condition": "动作结束条件"
}
工具参数：
- move_ee_to_grounded_instance：arm、instance_id、action_mode=grasp/contact/place、point_key=approach_world_m/grasp_world_m/contact_world_m/place_world_m、max_translation、steps。目标来自 scene_memory，target_quat_wxyz=grounded 使用对应方向；offset_xyz 是以米计的世界坐标偏移。
- place 的必需字段为 arm、action_mode="place"、target_id、point_key；target_id 从 operation_targets 复制。instance_id 若提供，指正在携带的物体，不能填写放置目标的 target_id。先使用 approach_world_m，再使用 place_world_m；放置几何已经包含物体与 TCP 的相对关系及接近高度，place 中省略 preserve_height 或设置为 false。
- 直接搬运到某个物体上方时可使用 preserve_height=true；这种调用使用目的物体的 instance_id 及正在携带物体的 held_instance_id。此方式与 action_mode="place" 的目标位姿接口分别使用。
- move_ee_to_pose：arm、来自输入的 target_pose=[x,y,z,w,qx,qy,qz]，或者 target_xyz=[x,y,z] 与可选 target_quat_wxyz=[w,qx,qy,qz]，可附 max_translation、steps。坐标和四元数为数值数组，不能传入字典。仅改变位置时省略 target_quat_wxyz 以保持当前姿态。
- lift_ee：arm、distance（最多 0.05 米）、steps，用于任务需要的局部抬升。
- open_gripper/close_gripper：arm。接近点与抓取点分别对应 approach_world_m 和 grasp_world_m。
- contact_displace：arm、axis=x/y/z、direction=positive/negative、distance（最多 0.04 米）、steps。方向匹配目标的接触几何。
- 瞬时接触设置 complete_transient_cycle=true；同批提供 contact 模式的目标动作，Runtime 追加返回该目标接近点的释放动作。
- 运动可携带 gripper_precondition=open/closed。max_translation 限制每个内部步长；steps 为内部控制步数，最大 20。local_grounded_goal_control=true 时 Runtime 会在该上限内继续完成目标。
空 tool_calls 配合 replan 请求重新规划；abort 结束本次执行。
""" + DIRECT_FEEDBACK_INSTRUCTIONS


def update_commanded_manipulation(agent: Any, calls: list, results: list) -> None:
    working = agent.memory_store.state.working
    previous = working.manipulation_state
    updated = {arm: dict(state) for arm, state in previous.items()}
    setups = agent._commanded_grasp_setups
    step = int(agent.latest_snapshot.step_count)
    for call, result in zip(calls, results):
        details = dict(result.details or {})
        if not result.success or details.get("skipped"):
            continue
        arm = agent._single_arm_from_call(call)
        if call.tool_name == "open_gripper":
            for selected in (("left", "right") if arm == "both" else (arm,)):
                updated.pop(selected, None)
                setups.pop(selected, None)
            continue
        if arm not in {"left", "right"}:
            continue
        if call.tool_name == "move_ee_to_grounded_instance":
            setups.pop(arm, None)
            mode = details.get("operation_action_mode", call.args.get("_operation_action_mode"))
            if mode == "grasp":
                if (
                    agent._normalized_grounded_point_key(call) in {"grasp_world_m", "contact_world_m"}
                    and details.get("target_reached") is True
                ):
                    setups[arm] = call
            continue
        if call.tool_name != "close_gripper":
            setups.pop(arm, None)
            continue
        if arm not in setups:
            continue
        setup = setups.pop(arm)
        instance = agent._grounded_instance_for_call(setup)
        if not isinstance(instance, dict):
            continue
        instance_id = agent._recovery_history_instance_id(setup, {})
        object_world = agent._xyz_prefix(instance.get("latest_world_m", instance.get("world_m")))
        pose = agent._pose7_prefix(details.get("observed_pose"))
        if pose is None:
            pose = agent._pose7_prefix((details.get("observed_poses_by_arm") or {}).get(arm))
        robot_arm = agent._get_robot_state().get(arm) if pose is None else {"xyz": pose[:3], "quat_wxyz": pose[3:]}
        attachment = capture_held_object_to_tcp_attachment(
            object_world_m=object_world,
            robot_arm_state=robot_arm,
            calibration=agent.latest_snapshot.tcp_calibration_by_arm.get(arm),
        )
        if attachment is not None:
            attachment["source"] = "executed_gripper_command"
        # 保留坐标变换供后续移动使用；抓取状态仅来自已经执行的闭合命令。
        updated[arm] = {
            "phase": "holding_assumed",
            "state_source": "executed_gripper_command",
            "execution_evidence_enabled": False,
            "held_instance_id": instance_id,
            "holding_confirmed": False,
            "transport_authorized": attachment is not None,
            "held_object_to_tcp_attachment": attachment,
            "pregrasp_object_world_m": object_world,
            "held_object_perception_descriptor": agent._perception_descriptor_for_scene_instance(instance),
            "source_skill_id": agent._active_skill_id(),
            "updated_step": step,
        }
    working.manipulation_state = updated
    agent._sync_runtime_manipulation_state_to_scene_memory()
    agent._record_trace_and_rollout_event("manipulation_command_update", {
        "previous": previous,
        "current": updated,
        "execution_evidence_enabled": False,
    })


def resume_planning_from_feedback(agent: Any) -> None:
    agent.memory_store.clear_active_skill()
    agent._debug_recovery_triggered = False
    agent._debug_recovery_planner_bootstrapped = False
    agent._debug_recovery_rounds = 0
    agent._debug_recovery_scene_wait_turns = 0
    agent._pure_tool_control_empty_plan_turns = 0
    agent.memory_store.set_monitor_status(
        phase="reasoning", status="needs_reasoning",
        note="Plan from the latest images and executed action feedback.",
        env_signal="running", failure_reason="", progress_score=0.0,
    )
    agent._record_trace_and_rollout_event("subtask_transition_decision", {
        "authority": "planner",
        "execution_evidence_enabled": False,
        "resolved_control": "replan",
    })
