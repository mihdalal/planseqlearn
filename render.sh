python planseqlearn/generate_video_from_states_non_robosuite.py --env_name SawyerAssemblyObstacle-v0 --camera_name wrist --suite mopa
python planseqlearn/generate_video_from_states_non_robosuite.py --env_name SawyerPushObstacle-v0 --camera_name wrist --suite mopa
python planseqlearn/generate_video_from_states_non_robosuite.py --env_name SawyerLiftObstacle-v0 --camera_name wrist --suite mopa

python planseqlearn/generate_video_from_states_non_robosuite.py --env_name kitchen-microwave-v0 --camera_name wrist --suite kitchen
python planseqlearn/generate_video_from_states_non_robosuite.py --env_name kitchen-kettle-v0 --camera_name wrist --suite kitchen
python planseqlearn/generate_video_from_states_non_robosuite.py --env_name kitchen-tlb-v0 --camera_name wrist --suite kitchen
python planseqlearn/generate_video_from_states_non_robosuite.py --env_name kitchen-light-v0 --camera_name wrist --suite kitchen
# python planseqlearn/generate_video_from_states_non_robosuite.py --env_name kitchen-slide-v0 --camera_name wrist --suite kitchen
# python planseqlearn/generate_video_from_states_non_robosuite.py --env_name kitchen-ms3-v0 --camera_name wrist --suite kitchen
python planseqlearn/generate_video_from_states_non_robosuite.py --env_name kitchen-ms5-v0 --camera_name wrist --suite kitchen
# python planseqlearn/generate_video_from_states_non_robosuite.py --env_name kitchen-ms7-v0 --camera_name wrist --suite kitchen
python planseqlearn/generate_video_from_states_non_robosuite.py --env_name kitchen-ms10-v0 --camera_name wrist --suite kitchen

python planseqlearn/generate_video_from_states_non_robosuite.py --env_name bin-picking-v2 --camera_name gripperPOVpos --suite metaworld
python planseqlearn/generate_video_from_states_non_robosuite.py --env_name assembly-v2 --camera_name gripperPOVneg --suite metaworld
python planseqlearn/generate_video_from_states_non_robosuite.py --env_name disassemble-v2 --camera_name gripperPOVneg --suite metaworld
python planseqlearn/generate_video_from_states_non_robosuite.py --env_name hammer-v2 --camera_name gripperPOVpos --suite metaworld

python planseqlearn/generate_video_from_states.py --env_name PickPlaceCan --camera_name robot0_eye_in_hand
python planseqlearn/generate_video_from_states.py --env_name PickPlaceBread --camera_name robot0_eye_in_hand
python planseqlearn/generate_video_from_states.py --env_name PickPlaceMilk --camera_name robot0_eye_in_hand
python planseqlearn/generate_video_from_states.py --env_name PickPlaceCereal --camera_name robot0_eye_in_hand
python planseqlearn/generate_video_from_states.py --env_name PickPlaceCerealMilk --camera_name robot0_eye_in_hand
python planseqlearn/generate_video_from_states.py --env_name PickPlaceCanBread --camera_name robot0_eye_in_hand
python planseqlearn/generate_video_from_states.py --env_name Lift --camera_name robot0_eye_in_hand
# python planseqlearn/generate_video_from_states.py --env_name Door --camera_name robot0_eye_in_hand
python planseqlearn/generate_video_from_states.py --env_name NutAssembly --camera_name robot0_eye_in_hand
python planseqlearn/generate_video_from_states.py --env_name NutAssemblyRound --camera_name robot0_eye_in_hand
python planseqlearn/generate_video_from_states.py --env_name NutAssemblySquare --camera_name robot0_eye_in_hand