using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
const rbd = RigidBodyDynamics
using LearningMPC.Models
using LCMCore
using BotCoreLCMTypes: robot_state_t
using StaticArrays
using Rotations

robot = BoxAtlas()
mech = mechanism(robot)
atlas_world_frame = CartesianFrame3D("atlas_world")
add_frame!(root_body(mech), Transform3D(atlas_world_frame, default_frame(root_body(mech)), RotZ(π/2)))

state = MechanismState(mech)

vis = Visualizer()
mvis = MechanismVisualizer(robot, vis)
open(mvis)
wait(mvis)

lcm = LCM()
function on_robot_state(channel::String, msg::robot_state_t)
    com_position = Point3D(atlas_world_frame, msg.pose.translation.x, msg.pose.translation.y, msg.pose.translation.z)
    
    world = default_frame(root_body(mech))
    pcom = relative_transform(state, atlas_world_frame, world) * com_position
    set_configuration!(state, findjoint(mech, "base_x"), pcom.v[1])
    set_configuration!(state, findjoint(mech, "base_z"), pcom.v[3])

    for (body, idxs, sign, θ) in (("lf", 7:9, 1, π), ("rf", 10:12, -1, π), ("lh", 1:3, 1, π/2), ("rh", 4:6, -1, π/2))
        position = Point3D(atlas_world_frame, msg.joint_position[idxs])
        rotation_joint = findjoint(mech, "core_to_$(body)_rotation")
        offset = relative_transform(state, atlas_world_frame, frame_before(rotation_joint)) * position
        set_configuration!(state, rotation_joint, θ + sign * atan2(offset.v[1], offset.v[3]))

        extension_joint = findjoint(mech, "core_to_$(body)_extension")
        offset = relative_transform(state, atlas_world_frame, frame_before(extension_joint)) * position
        set_configuration!(state, extension_joint, offset.v' * extension_joint.joint_type.axis)
    end
    set_configuration!(mvis, configuration(state))
end

subscribe(lcm, "BOX_ATLAS_STATE", on_robot_state, robot_state_t)
while true
    handle(lcm)
end
