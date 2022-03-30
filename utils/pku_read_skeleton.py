import numpy as np
import os


def read_skeleton(file):
    #print(file)
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            # frame_info['numBody'] = int(float(f.readline()))

            numBody = f.readline()
            try:
                frame_info['numBody'] = int(float(numBody))
            except:
                frame_info['numBody'] = 's'
                print (file)
                input()
                return None
                
            # if not isinstance(frame_info['numBody'], int):
            #     print (frame_info['numBody'])
            #     print (file)
            #     input()

            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                if m==0:
                    body_info_key = [
                        'bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isResticted', 'leanX', 'leanY', 'trackingState'
                    ]
                    body_info = {
                        k: float(v)
                        for k, v in zip(body_info_key, f.readline().split())
                    }

                    body_info['numJoint'] = int(f.readline())
                else:
                    body_info_key = [
                        'bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isResticted', 'leanX', 'leanY', 'trackingState'
                    ]
                    body_info = {
                        k: float(v)
                        for k, v in zip(body_info_key, '6 1 0 0 1 1 0 -0.437266 -0.117168 2'.split())
                    }
                    body_info['numJoint'] = 25
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z'#, 'depthX', 'depthY', 'colorX', 'colorY',
                        #'orientationW', 'orientationX', 'orientationY',
                        #'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    if seq_info is None:
        return np.zeros((3, 300, num_joint, max_body))
    if seq_info['numFrame'] > 300:
        start = (seq_info['numFrame'] - 300) // 2
        end = 300 + (seq_info['numFrame'] - 300) // 2
        seq_info['frameInfo'] = [f for n, f in enumerate(seq_info['frameInfo']) if n in range(start,end)]
        seq_info['numFrame'] = 300
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data
