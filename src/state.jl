using RigidBodyDynamics
using BotCoreLCMTypes: robot_state_t
using StaticArrays: SVector
using Rotations: RodriguesVec, RotZ, Quat
using LearningMPC.Models: BoxAtlas, mechanism, environment
using LCPSim

struct JointIKParams{T}
    body::RigidBody{T}
    indices::Vector{Int}
    rotation_joint::Joint{T, Revolute{T}}
    extension_joint::Joint{T, Prismatic{T}}
    rotation_offset::Float64
end

struct BoxAtlasStateTranslator{T, M <: MechanismState}
    robot::BoxAtlas{T}
    state::M
    atlas_world_frame::CartesianFrame3D
    bodies::Dict{String, RigidBody{T}}
    ik_params::Vector{JointIKParams{T}}
    floating_base::Joint{T, Planar{T}}
end

function BoxAtlasStateTranslator(robot::BoxAtlas{T}) where T
    mech = mechanism(robot)
    state = MechanismState(mech)
    atlas_world_frame = CartesianFrame3D("atlas_world")
    add_frame!(root_body(mech), Transform3D(atlas_world_frame, default_frame(root_body(mech)), RotZ(0.0)))
    bodies = Dict([s => findbody(mech, s) for s in ["core", "lf", "rf", "lh", "rh"]])
    ik_data = [("lf", 7:9, π), ("rf", 10:12, π), ("lh", 1:3, π/2), ("rh", 4:6, π/2)]
    ik_params = map(ik_data) do x
        name, idx, offset = x
        body = bodies[name]
        JointIKParams{T}(
            body, 
            idx, 
            findjoint(mech, "core_to_$(name)_rotation"),
            findjoint(mech, "core_to_$(name)_extension"),
            offset)
    end
    floating_base = findjoint(mech, "floating_base")
    BoxAtlasStateTranslator(robot, state, atlas_world_frame, bodies, ik_params, floating_base)
end

function _update_floating_base!(state::MechanismState, translator::BoxAtlasStateTranslator, msg::robot_state_t)
    core_frame = default_frame(translator.bodies["core"])
    atlas_world_frame = translator.atlas_world_frame
    mech = mechanism(translator.robot)

    com_pose = Transform3D(core_frame, atlas_world_frame, 
                           Quat(msg.pose.rotation.w, msg.pose.rotation.x, msg.pose.rotation.y, msg.pose.rotation.z),
                           SVector(msg.pose.translation.x, msg.pose.translation.y, msg.pose.translation.z))
    com_twist = Twist(core_frame, 
                      atlas_world_frame, 
                      FreeVector3D(atlas_world_frame, msg.twist.angular_velocity.x, msg.twist.angular_velocity.y, msg.twist.angular_velocity.z),
                      FreeVector3D(atlas_world_frame, msg.twist.linear_velocity.x, msg.twist.linear_velocity.y, msg.twist.linear_velocity.z))

    world = default_frame(root_body(mech))
    floating_base = translator.floating_base
    Hcom = relative_transform(state, atlas_world_frame, world) * com_pose
    rcom = RodriguesVec(rotation(Hcom))
    H = relative_transform(state, frame_before(floating_base), world)
    xaxis = H * FreeVector3D(frame_before(floating_base), floating_base.joint_type.x_axis)
    yaxis = H * FreeVector3D(frame_before(floating_base), floating_base.joint_type.y_axis)
    set_configuration!(state, floating_base, SVector(translation(Hcom)' * xaxis.v, 
                                                     translation(Hcom)' * yaxis.v, 
                                                     SVector(rcom.sx, rcom.sy, rcom.sz)' * floating_base.joint_type.rot_axis))

    Tcom = transform(com_twist, relative_transform(state, atlas_world_frame, world))
    set_velocity!(state, floating_base, SVector(linear(Tcom)' * floating_base.joint_type.x_axis, 
                                                linear(Tcom)' * floating_base.joint_type.y_axis,
                                                angular(Tcom)' * floating_base.joint_type.rot_axis))
end

function _update_limb!(state::MechanismState, 
                       ik_params::JointIKParams, 
                       atlas_world_frame::CartesianFrame3D, 
                       world_frame::CartesianFrame3D,
                       msg::robot_state_t)
    position = Point3D(atlas_world_frame, msg.joint_position[ik_params.indices])
    rotation_joint = ik_params.rotation_joint
    @assert rotation_joint.joint_type.axis == SVector(0, 1, 0) || rotation_joint.joint_type.axis == SVector(0, -1, 0)
    extension_joint = ik_params.extension_joint

    H = relative_transform(state, atlas_world_frame, frame_before(rotation_joint))
    offset = H * position
    θ = ik_params.rotation_offset + rotation_joint.joint_type.axis[2] * atan2(offset.v[1], offset.v[3])
    if θ > π
        θ -= 2π
    elseif θ < -π
        θ += 2π
    end
    set_configuration!(state, rotation_joint, θ)

    H = relative_transform(state, atlas_world_frame, frame_before(extension_joint))
    offset = H * position
    set_configuration!(state, extension_joint, offset.v' * extension_joint.joint_type.axis)

    H = relative_transform(state, atlas_world_frame, frame_before(rotation_joint))
    velocity = H * FreeVector3D(atlas_world_frame, msg.joint_velocity[ik_params.indices])
    twist_before_joint = transform(relative_twist(state, frame_before(rotation_joint), world_frame), 
                                   relative_transform(state, world_frame, frame_before(rotation_joint)))
    displacement = H * position - Point3D(frame_before(rotation_joint), 0, 0, 0)
    relative_velocity = velocity - FreeVector3D(twist_before_joint.frame, linear(twist_before_joint))
    set_velocity!(state, rotation_joint, 
                  cross(displacement, relative_velocity).v' * rotation_joint.joint_type.axis - angular(twist_before_joint)' * rotation_joint.joint_type.axis)

    axis = FreeVector3D(frame_before(extension_joint), extension_joint.joint_type.axis)
    set_velocity!(state, extension_joint, 
                  dot(relative_transform(state, frame_before(rotation_joint), frame_before(extension_joint)) * relative_velocity,
                      axis))
end

function update!(translator::BoxAtlasStateTranslator, msg::robot_state_t)
    _update_floating_base!(translator.state, translator, msg)
    for ik_params in translator.ik_params
        _update_limb!(translator.state, ik_params, translator.atlas_world_frame, root_frame(mechanism(translator.robot)), msg)
    end
end

function (t::BoxAtlasStateTranslator)(msg::robot_state_t)
    update!(t, msg)
    return t.state
end

struct OneStepSimulator{T <: BoxAtlasStateTranslator, C, S, M <: MechanismState}
    translator::T
    controller::C
    Δt::Float64
    lcp_solver::S
    state::M
end

function OneStepSimulator(t::BoxAtlasStateTranslator, controller, Δt, solver)
    state = MechanismState(mechanism(t.robot))
    OneStepSimulator(t, controller, Δt, solver, state)
end


function (s::OneStepSimulator)(state::MechanismState)
    world = root_frame(mechanism(s.translator.robot))
    H = relative_transform(state, world, s.translator.atlas_world_frame)
    positions = Dict(map(values(s.translator.bodies)) do body
        body => [H * Point3D(world, translation(transform_to_root(state, body)))]
    end)
    velocities = Dict(map(values(s.translator.bodies)) do body
        body => [H * FreeVector3D(world, linear(twist_wrt_world(state, body)))]
    end)

    set_configuration!(s.translator.state, configuration(state))
    set_velocity!(s.translator.state, velocity(state))
    results = LCPSim.simulate(s.translator.state, s.controller, environment(s.translator.robot), s.Δt, 1, s.lcp_solver)
    for result in results
        set_configuration!(s.translator.state, configuration(result.state))
        set_velocity!(s.translator.state, velocity(result.state))
        H = relative_transform(s.translator.state, world, s.translator.atlas_world_frame)
        for body in keys(positions)
            push!(positions[body], H * Point3D(world, translation(transform_to_root(s.translator.state, body))))
            push!(velocities[body], H * FreeVector3D(world, linear(twist_wrt_world(s.translator.state, body))))
        end
    end

    splines = Dict(map(values(s.translator.bodies)) do body
        times = vcat(0.0, cumsum([s.Δt for r in results]))
        for p in positions[body]
            @assert p.frame === s.translator.atlas_world_frame
        end
        for v in velocities[body]
            @assert v.frame === s.translator.atlas_world_frame
        end
        body => map(1:3) do i
            p = [x.v[i] for x in positions[body]]
            v = [x.v[i] for x in velocities[body]]
            cubic_spline(times, p, v)
        end
    end)
    @show splines[s.translator.bodies["core"]]
    return splines
end


