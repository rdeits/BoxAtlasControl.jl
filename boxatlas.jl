using Colors
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

floating_base = findjoint(mech, "floating_base")
floating_x = joint_type(floating_base).x_axis
floating_y = joint_type(floating_base).y_axis
floating_rot = joint_type(floating_base).rot_axis

state = MechanismState(mech)

vis = Visualizer()
mvis = MechanismVisualizer(robot, vis)
open(mvis)
wait(mvis)

lcm = LCM()

draw_velocity!(mvis::MechanismVisualizer, state::MechanismState, name::AbstractString) = 
    draw_velocity!(mvis, state, findbody(state.mechanism, name))

function draw_velocity!(mvis::MechanismVisualizer, state::MechanismState, body::RigidBody)
    vis = mvis.visualizer["velocities"][string(body)]
    pose = transform_to_root(state, body)
    twist = twist_wrt_world(state, body)
    setobject!(vis, LineSegments(PointCloud([translation(pose), translation(pose) + linear(twist)]),
                                 MeshCat.LineBasicMaterial(color=colorant"red")))
end


function on_robot_state(channel::String, msg::robot_state_t)
    com_pose = Transform3D(core_frame, atlas_world_frame, 
                           Quat(msg.pose.rotation.w, msg.pose.rotation.x, msg.pose.rotation.y, msg.pose.rotation.z),
                           SVector(msg.pose.translation.x, msg.pose.translation.y, msg.pose.translation.z))

    com_twist = Twist(core_frame, 
                      atlas_world_frame, 
                      FreeVector3D(atlas_world_frame, msg.twist.angular_velocity.x, msg.twist.angular_velocity.y, msg.twist.angular_velocity.z),
                      FreeVector3D(atlas_world_frame, msg.twist.linear_velocity.x, msg.twist.linear_velocity.y, msg.twist.linear_velocity.z))
    
    world = default_frame(root_body(mech))
    Hcom = relative_transform(state, atlas_world_frame, world) * com_pose
    rcom = RodriguesVec(rotation(Hcom))
    set_configuration!(state, floating_base, SVector(translation(Hcom)' * floating_x, 
                                                     translation(Hcom)' * floating_y, 
                                                     SVector(rcom.sx, rcom.sy, rcom.sz)' * floating_rot))

    Tcom = transform(com_twist, relative_transform(state, atlas_world_frame, world))
    set_velocity!(state, floating_base, SVector(linear(Tcom)' * floating_x, 
                                                linear(Tcom)' * floating_y,
                                                angular(Tcom)' * floating_rot))

    for (body, idxs, sign, θ) in (("lf", 7:9, 1, π), ("rf", 10:12, -1, π), ("lh", 1:3, 1, π/2), ("rh", 4:6, -1, π/2))
        position = Point3D(atlas_world_frame, msg.joint_position[idxs])
        rotation_joint = findjoint(mech, "core_to_$(body)_rotation")
        H = relative_transform(state, atlas_world_frame, frame_before(rotation_joint))
        offset = H * position
        set_configuration!(state, rotation_joint, θ + sign * atan2(offset.v[1], offset.v[3]))

        extension_joint = findjoint(mech, "core_to_$(body)_extension")
        H = relative_transform(state, atlas_world_frame, frame_before(extension_joint))
        offset = H * position
        set_configuration!(state, extension_joint, offset.v' * extension_joint.joint_type.axis)

        H = relative_transform(state, atlas_world_frame, frame_before(rotation_joint))
        velocity = H * FreeVector3D(atlas_world_frame, msg.joint_velocity[idxs])
        twist_before_joint = transform(relative_twist(state, frame_before(rotation_joint), world), 
                                       relative_transform(state, world, frame_before(rotation_joint)))
        displacement = H * position - Point3D(frame_before(rotation_joint), 0, 0, 0)
        relative_velocity = velocity - FreeVector3D(twist_before_joint.frame, linear(twist_before_joint))
        set_velocity!(state, rotation_joint, 
                      cross(displacement, relative_velocity).v' * rotation_joint.joint_type.axis - angular(twist_before_joint)' * rotation_joint.joint_type.axis)

        axis = FreeVector3D(frame_before(extension_joint), extension_joint.joint_type.axis)
        set_velocity!(state, extension_joint, 
                      dot(relative_transform(state, frame_before(rotation_joint), frame_before(extension_joint)) * relative_velocity,
                          axis))
    end

    draw_velocity!.(mvis, state, ["lf", "rf", "lh", "rh", "core"])

    set_configuration!(mvis, configuration(state))
end

subscribe(lcm, "BOX_ATLAS_STATE", on_robot_state, robot_state_t)
while true
    handle(lcm)
end
