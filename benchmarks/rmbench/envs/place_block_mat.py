from ._base_task import Base_Task
from .utils import *

class place_block_mat(Base_Task):
    
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.block_half_size = 0.02
        def create_block(block_pose):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=(1,0,0),
                name="box",
            )
        
        mat_half_size = [0.04, 0.04, 0.0005]
        blue_color = (0.000, 0.502, 0.996)
        green_color = (0.044, 0.601, 0.395)
        def create_mat(color, mat_pose):
            return create_box(
                scene=self,
                pose=mat_pose,
                half_size=mat_half_size,
                color=color,
                name="box",
                is_static=True,
            )
        
        mats_pose = []
        x_mat = -0.3
        for _ in range(6):
            mat_pos = rand_pose(
                xlim=[x_mat, x_mat],
                ylim=[-0.1, -0.1],
                qpos=[1, 0, 0, 0],
            )
            mats_pose.append(mat_pos)
            x_mat += 0.12
        self.mat_lst=[]
        for i in range(6):
            mat = create_mat(blue_color, mats_pose[i])
            self.mat_lst.append(mat)
        
        self.green_pos = rand_pose(
                xlim=[0, 0],
                ylim=[-0.25, -0.25],
                qpos=[1, 0, 0, 0],
            )
        self.green_mat = create_mat(green_color, self.green_pos)
        
        mats_idx = np.array([0, 1, 2, 3, 4, 5])
        self.selected_idx = np.sort(np.random.choice(mats_idx, size=3, replace=False))
        self.blocks_lst = []
        for i in range(3):
            block_idx = self.selected_idx[i]
            block_pose = mats_pose[block_idx]
            block = create_block(block_pose)
            self.blocks_lst.append(block)

        self.flag = ['False', 'False', 'False']
    
    def play_once(self):
        for i in range(3):
            self.place_back_block(i)
        self.info["info"] = {}
        return self.info
        
    def place_back_block(self, block_idx):
        block_name = ["left", "middle", "right"][block_idx]
        mat_idx = self.selected_idx[block_idx]
        arm_tag = self.choose_arm(mat_idx)
        self.move(self.grasp_actor(self.blocks_lst[block_idx], arm_tag=arm_tag, pre_grasp_dis = 0.1, grasp_dis=0.02), language_annotation=f'Pick up the {block_name} block and move it onto the green mat.')
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f'Pick up the {block_name} block and move it onto the green mat.')
        self.move(self.place_actor(self.blocks_lst[block_idx], arm_tag=arm_tag, target_pose=self.green_mat.get_pose().p, functional_point_id=2, dis=0.01), language_annotation=f'Pick up the {block_name} block and move it onto the green mat.')
        self.check_block_on_green_mat(block_idx)
        self.move(self.back_to_origin(arm_tag=arm_tag), language_annotation=f'Pick up the {block_name} block and move it onto the green mat.')
        self.move(self.grasp_actor(self.blocks_lst[block_idx], arm_tag=arm_tag, pre_grasp_dis = 0.1, grasp_dis=0.02), language_annotation="Pick up the block on the green mat and put it back to the original mat.")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1), language_annotation=f'Pick up the block on the green mat and put it back to the original mat.')    
        self.move(self.place_actor(self.blocks_lst[block_idx], arm_tag=arm_tag, target_pose=self.mat_lst[mat_idx].get_pose().p, functional_point_id=2, dis=0.01), language_annotation=f'Pick up the block on the green mat and put it back to the original mat.')
        self.move(self.back_to_origin(arm_tag=arm_tag), language_annotation=f'Pick up the block on the green mat and put it back to the original mat.')

    def choose_arm(self, block_idx):
        if block_idx <= 2:
            arm = "left"
        elif 2 < block_idx <= 5:
            arm = "right"
        return arm
    
    def check_block_on_green_mat(self, block_idx):
        block_pose = self.blocks_lst[block_idx].get_pose().p
        green_pose = self.green_mat.get_pose().p
        if np.abs(block_pose[0] - green_pose[0]) < 0.03 and np.abs(block_pose[1] - green_pose[1]) < 0.03 and block_pose[2] < 0.77:
            self.flag[block_idx] = 'True'

    def check_success(self):
        for i in range(3):
            self.check_block_on_green_mat(i)
            if self.flag[i] != 'True':
                return False
        for i in range(3):
            block_pose = self.blocks_lst[i].get_pose().p
            mat_idx = self.selected_idx[i]
            mat_pose = self.mat_lst[mat_idx].get_pose().p
            if np.abs(block_pose[0] - mat_pose[0]) >= 0.03 or np.abs(block_pose[1] - mat_pose[1]) >= 0.03 or block_pose[2] >= 0.77:
                return False
        return True