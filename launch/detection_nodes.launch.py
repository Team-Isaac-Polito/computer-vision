from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 獲取 realsense2_camera 包的路徑
    try:
        realsense_pkg_dir = get_package_share_directory('realsense2_camera')
        realsense_launch_file = os.path.join(realsense_pkg_dir, 'launch', 'rs_launch.py')
    except Exception as e:
        # 如果找不到包，則將路徑設為空
        print(f"Warning: realsense2_camera package not found: {e}")
        realsense_launch_file = None

    # 1. 啟動 RealSense 相機 (如果路徑找到)
    # ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
    realsense_camera_launch = []
    if realsense_launch_file:
        realsense_camera_launch.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments={
                    'align_depth.enable': 'true', 
                }.items()
            )
        )
    
    # 2. 啟動偵測管理器節點 (ros2 run reseq_ros2 reseq_ros2.detection_manager)
    detection_manager_node = Node(
        package='reseq_ros2',
        executable='reseq_ros2.detection_manager',
        name='detection_manager',
        output='screen'
    )

    # 3. 啟動偵測器節點 (ros2 run reseq_ros2 detector)
    detector_node = Node(
        package='reseq_ros2',
        executable='detector', 
        name='detector',
        output='screen'
    )
    
    # 4. 服務呼叫 (設置模式為 2)
    # 這裡使用 'sleep 2' 確保節點有時間啟動並提供服務
    set_mode_service_call = ExecuteProcess(
        cmd=['sleep', '2', '&&', 
             'ros2', 'service', 'call', '/detection/set_mode', 
             'reseq_interfaces/srv/SetMode', '"{mode: 2}"'],
        output='screen',
        shell=True,
    )

    return LaunchDescription(
        realsense_camera_launch + [
            detection_manager_node,
            detector_node,
            set_mode_service_call
        ]
    )