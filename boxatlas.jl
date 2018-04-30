using Revise

using MeshCat
using RigidBodyDynamics
const rbd = RigidBodyDynamics

using Colors: @colorant_str
using MeshCatMechanisms: MechanismVisualizer, URDFVisuals
using LearningMPC: mechanism, environment, LQRSolution, MPCParams
using LearningMPC.Models: BoxAtlas
using LCMCore: LCM, subscribe, handle, set_queue_capacity, publish
using BotCoreLCMTypes: robot_state_t
using DrakeLCMTypes: whole_body_data_t, piecewise_polynomial_t, qp_controller_input_t
using LCMPolynomials: cubic_spline
using BoxAtlasControl: BoxAtlasStateTranslator, OneStepSimulator, LIPM_zmp_data, body_motion_data, foot_support_data
using StaticArrays: SVector
using MAT: matopen
using AtlasRobot

draw_velocity!(mvis::MechanismVisualizer, state::MechanismState, name::AbstractString) = 
    draw_velocity!(mvis, state, findbody(state.mechanism, name))

function draw_velocity!(mvis::MechanismVisualizer, state::MechanismState, body::RigidBody)
    vis = mvis.visualizer["velocities"][string(body)]
    pose = transform_to_root(state, body)
    twist = twist_wrt_world(state, body)
    setobject!(vis, LineSegments(PointCloud([translation(pose), translation(pose) + linear(twist)]),
                                 MeshCat.LineBasicMaterial(color=colorant"red")))
end

robot = BoxAtlas()
translator = BoxAtlasStateTranslator(robot)

vis = Visualizer()
mvis = MechanismVisualizer(robot, vis[:boxatlas])
open(mvis)
wait(mvis)

mpc_params = LearningMPC.MPCParams(robot)
mpc_params.horizon = 1
mpc_params.Δt = 0.01
lqrsol = LearningMPC.LQRSolution(robot, mpc_params)

vis_handler = function(state)
    set_configuration!(mvis, configuration(state))
    for body in values(translator.bodies)
        draw_velocity!.(mvis, state, body)
    end
    state
end

sim = OneStepSimulator(translator, lqrsol, mpc_params.Δt, mpc_params.lcp_solver)

function qp_input_generator(state, splines, timestamp=0)
    msg = qp_controller_input_t()
    msg.timestamp = timestamp
    msg.be_silent = false
    msg.zmp_data = LIPM_zmp_data(SVector(0., 0, 0, 0), SVector(0., 0))
    msg.num_support_data = 2
    msg.support_data = [
        foot_support_data("r_foot"),
        foot_support_data("l_foot")
    ]

    now = timestamp / 1e6

    msg.num_tracked_bodies = length(splines)
    msg.body_motion_data = [body_motion_data(string(body), spline) for (body, spline) in splines]

    msg.num_external_wrenches = 0

    fname = joinpath(@__DIR__, "data", "atlas_fp.mat")
    fp_file = matopen(fname)
    xstar = try
        read(fp_file, "xstar")
    finally
        close(fp_file)
    end
    qstar = xstar[1:36]
    q_spline = [cubic_spline(SVector(now, now + 1), SVector(qi, qi), SVector(0., 0)) for qi in qstar]
    msg.whole_body_data = whole_body_data_t(
        timestamp,
        36,
        qstar,
        piecewise_polynomial_t(q_spline),
        17,
        [10, 17, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 36, 8, 7]
    )

    msg.num_joint_pd_overrides = 0
    msg.param_set_name = "standing"
    msg.torque_alpha_filter = 0
    msg
end

lcm = LCM()

function handle_box_atlas_state(channel, msg::robot_state_t)
    state = translator(msg)
    vis_handler(state)
    splines = sim(state)
    qp_input_msg = qp_input_generator(state, splines, msg.utime)
    publish(lcm, "QP_CONTROLLER_INPUT", qp_input_msg)
end

sub = subscribe(lcm, "BOX_ATLAS_STATE", handle_box_atlas_state, robot_state_t)
set_queue_capacity(sub, 1)

atlas = AtlasRobot.mechanism()
atlas_vis = MechanismVisualizer(atlas, URDFVisuals(AtlasRobot.urdfpath(), package_path=[AtlasRobot.packagepath()]), vis[:atlas])

using HumanoidLCMSim.AtlasSim: atlasrobotinfo
using HumanoidLCMSim: set!

atlas_info = atlasrobotinfo(atlas)
atlas_state = MechanismState(atlas)

function handle_atlas_state(channel, msg::robot_state_t)
    set!(atlas_state, msg, atlas_info)
    set_configuration!(atlas_vis, configuration(atlas_state))
end

sub = subscribe(lcm, "EST_ROBOT_STATE", handle_atlas_state, robot_state_t)
set_queue_capacity(sub, 1)

finished = Ref(false)

@async begin
    while !finished[]
        handle(lcm)
    end
end
