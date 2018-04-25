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
core_frame = default_frame(findbody(mech, "core"))

state = MechanismState(mech)

vis = Visualizer()
mvis = MechanismVisualizer(robot, vis)
open(mvis)
wait(mvis)

lcm = LCM()
function on_robot_state(channel::String, msg::robot_state_t)
    try
        com_pose = Transform3D(core_frame, atlas_world_frame, 
                               Quat(msg.pose.rotation.w, msg.pose.rotation.x, msg.pose.rotation.y, msg.pose.rotation.z),
                               SVector(msg.pose.translation.x, msg.pose.translation.y, msg.pose.translation.z))

        com_twist = Twist(core_frame, 
                          atlas_world_frame, 
                          FreeVector3D(atlas_world_frame, msg.twist.angular_velocity.x, msg.twist.angular_velocity.y, msg.twist.angular_velocity.z),
                          FreeVector3D(atlas_world_frame, msg.twist.linear_velocity.x, msg.twist.linear_velocity.y, msg.twist.linear_velocity.z))
        
        world = default_frame(root_body(mech))
        Tcom = relative_transform(state, atlas_world_frame, world) * com_pose
        pcom = translation(Tcom)
        rcom = RodriguesVec(rotation(Tcom))
        set_configuration!(state, findjoint(mech, "floating_base"), [pcom[1], pcom[3], rcom.sy])

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
    catch e
        @show e
    end

end

subscribe(lcm, "BOX_ATLAS_STATE", on_robot_state, robot_state_t)
while true
    handle(lcm)
end
