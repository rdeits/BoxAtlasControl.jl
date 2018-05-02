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
    bodies = Dict([s => findbody(mech, s) for s in ["pelvis", "l_foot_sole", "r_foot_sole", "l_hand_mount", "r_hand_mount"]])
    ik_data = [
        ("l_hand_mount", 1:3), 
        ("r_hand_mount", 4:6),
        ("l_foot_sole", 7:9), 
        ("r_foot_sole", 10:12), 
    ]
    ik_params = map(ik_data) do x
        name, idx = x
        body = bodies[name]
        JointIKParams{T}(
            body, 
            idx, 
            findjoint(mech, "pelvis_to_$(name)_rotation"),
            findjoint(mech, "pelvis_to_$(name)_extension"),
        )
    end
    floating_base = findjoint(mech, "floating_base")
    BoxAtlasStateTranslator(robot, state, atlas_world_frame, bodies, ik_params, floating_base)
end

function _update_floating_base!(state::MechanismState, translator::BoxAtlasStateTranslator, msg::robot_state_t)
    pelvis_frame = default_frame(translator.bodies["pelvis"])
    atlas_world_frame = translator.atlas_world_frame
    mech = mechanism(translator.robot)

    pelvis_pose = Transform3D(pelvis_frame, atlas_world_frame, 
                           Quat(msg.pose.rotation.w, msg.pose.rotation.x, msg.pose.rotation.y, msg.pose.rotation.z),
                           SVector(msg.pose.translation.x, msg.pose.translation.y, msg.pose.translation.z))
    pelvis_twist = Twist(pelvis_frame, 
                      atlas_world_frame, 
                      FreeVector3D(atlas_world_frame, msg.twist.angular_velocity.x, msg.twist.angular_velocity.y, msg.twist.angular_velocity.z),
                      FreeVector3D(atlas_world_frame, msg.twist.linear_velocity.x, msg.twist.linear_velocity.y, msg.twist.linear_velocity.z))

    world = default_frame(root_body(mech))
    floating_base = translator.floating_base
    Hcom = relative_transform(state, atlas_world_frame, world) * pelvis_pose 
    rcom = RodriguesVec(rotation(Hcom))
    H = relative_transform(state, frame_before(floating_base), world)
    xaxis = H * FreeVector3D(frame_before(floating_base), floating_base.joint_type.x_axis)
    yaxis = H * FreeVector3D(frame_before(floating_base), floating_base.joint_type.y_axis)
    set_configuration!(state, floating_base, SVector(translation(Hcom)' * xaxis.v, 
                                                     translation(Hcom)' * yaxis.v, 
                                                     SVector(rcom.sx, rcom.sy, rcom.sz)' * floating_base.joint_type.rot_axis))

    Tcom = transform(pelvis_twist, relative_transform(state, atlas_world_frame, world))
    set_velocity!(state, floating_base, SVector(linear(Tcom)' * xaxis.v, 
                                                linear(Tcom)' * yaxis.v,
                                                angular(Tcom)' * floating_base.joint_type.rot_axis))
end

function _update_limb!(state::MechanismState, 
                       ik_params::JointIKParams, 
                       base_frame::CartesianFrame3D, 
                       msg::robot_state_t)

    joint = ik_params.rotation_joint
    H = relative_transform(state, base_frame, frame_before(joint))
    relative_position = H * Point3D(base_frame, msg.joint_position[ik_params.indices])
    relative_velocity = H * FreeVector3D(base_frame, msg.joint_velocity[ik_params.indices])
    θ = joint.joint_type.axis[1] * atan2(-relative_position.v[2], relative_position.v[3])
    if θ > π
        θ -= 2π
    elseif θ < -π
        θ += 2π
    end
    set_configuration!(state, joint, θ)
    set_velocity!(state, joint, 
        dot(cross(relative_position, relative_velocity), 
            FreeVector3D(frame_before(joint), joint.joint_type.axis)))

    joint = ik_params.extension_joint
    H = relative_transform(state, base_frame, frame_before(joint))
    relative_position = H * Point3D(base_frame, msg.joint_position[ik_params.indices])
    relative_velocity = H * FreeVector3D(base_frame, msg.joint_velocity[ik_params.indices])
    set_configuration!(state, joint, 
        dot(FreeVector3D(relative_position), FreeVector3D(frame_before(joint), joint.joint_type.axis)))
    set_velocity!(state, joint, 
        dot(relative_velocity, FreeVector3D(frame_before(joint), joint.joint_type.axis)))
end

function update!(translator::BoxAtlasStateTranslator, msg::robot_state_t)
    _update_floating_base!(translator.state, translator, msg)
    pelvis_frame = default_frame(translator.bodies["pelvis"])
    for ik_params in translator.ik_params
        _update_limb!(translator.state, ik_params, pelvis_frame, msg)
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
    # @show splines[s.translator.bodies["pelvis"]]
    return splines
end


