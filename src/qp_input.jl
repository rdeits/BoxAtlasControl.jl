using LCMPolynomials
using HumanoidLCMSim
using LCMCore
using StaticArrays
using MAT: matopen
using BotCoreLCMTypes: robot_state_t
using DrakeLCMTypes: zmp_data_t,
                    support_data_t,
                    body_motion_data_t,
                    body_wrench_data_t,
                    whole_body_data_t,
                    joint_pd_override_t,
                    qp_controller_input_t

timestamp_now() = Int64(time() * 1e6)

function LIPM_zmp_data(x0=zeros(4), y0=zeros(2))
    zmp_data = zmp_data_t()
    zmp_data.A = @SMatrix [
        0 0 1 0;
        0 0 0 1;
        0 0 0 0;
        0 0 0 0
    ]
    zmp_data.B = @SMatrix [
        0 0;
        0 0;
        1 0;
        0 1;
    ]
    zmp_data.C = @SMatrix [
        1 0 0 0;
        0 1 0 0;
    ]
    zmp_data.D = @SMatrix [
        -0.101937 0;
        0 -0.101937;
    ]
    zmp_data.x0 = reshape(x0, 4, 1)
    zmp_data.y0 = reshape(y0, 2, 1)
    zmp_data.u0 = zeros(SMatrix{2, 1, Float64})
    zmp_data.R = zeros(SMatrix{2, 2, Float64})
    zmp_data.Qy = SDiagonal(1.0, 1.0)
    zmp_data.S = @SMatrix [
        0.6386 0      0.2039 0
        0      0.6386 0      0.2039
        0.2039 0      0.0651 0
        0      0.2039 0      0.0651
    ]
    zmp_data.s1 = zeros(SMatrix{4, 1, Float64})
    zmp_data.s1dot = zeros(SMatrix{4, 1, Float64})
    zmp_data.s2 = 0
    zmp_data.s2dot = 0
    zmp_data
end

function foot_support_data(name, surface=SVector(0., 0, 1, 0), mu=0.7)
    support_data_t(
        timestamp_now(),
        name,
        4,
        @SMatrix([0.1728    0.1728  -0.0876  -0.0876
                  0.0626   -0.0626   0.0626  -0.0626
                 -0.07645  -0.07645 -0.07645 -0.07645]),
        6000,
        5.0,
        ones(SVector{4, Bool}),
        true,
        surface,
        mu
    )
end

function body_motion_data(name, splines::AbstractVector{<:PiecewisePolynomial})
    msg = body_motion_data_t()
    msg.timestamp = timestamp_now()
    msg.body_or_frame_name = name
    msg.spline = piecewise_polynomial_t(splines)
    msg.quat_task_to_world = @SVector [1, 0, 0, 0]
    msg.translation_task_to_world = @SVector [0, 0, 0]
    msg.xyz_kp_multiplier = @SVector [1, 1, 1]
    msg.xyz_damping_ratio_multiplier = @SVector [1, 1, 1]
    msg.expmap_kp_multiplier = 1
    msg.expmap_damping_ratio_multiplier = 1
    msg.weight_multiplier = @SVector [1, 1, 1, 0, 0, 1]
    msg.in_floating_base_nullspace = false
    msg.control_pose_when_in_contact = false
    msg
end

function send_qp_controller_input()
    timestamp = Ref(0)
    LCM() do lcm
        handle_state(channel, msg) = timestamp[] = msg.utime
        sub = subscribe(lcm, "EST_ROBOT_STATE", handle_state, robot_state_t)
        handle(lcm)  # wait for a robot state
        unsubscribe(lcm, sub)
    end

    msg = qp_controller_input_t()
    msg.timestamp = timestamp[]
    msg.be_silent = false
    msg.zmp_data = LIPM_zmp_data(SVector(0., 0, 0, 0), SVector(0., 0))

    msg.num_support_data = 2
    msg.support_data = [
        foot_support_data("r_foot"),
        foot_support_data("l_foot")
    ]

    now = timestamp[] / 1e6

    msg.num_tracked_bodies = 1
    pelvis_coordinates = [0, 0, 0.8, 0, 0, 0]
    splines = [cubic_spline(now .+ SVector(0., 1, 2), SVector(c, c, c), SVector(0., 0, 0)) for c in pelvis_coordinates]
    splines[3] = cubic_spline(now .+ SVector(0, 1, 2), SVector(0.8, 0.9, 0.8), SVector(0., 0, 0))
    # @show splines
    msg.body_motion_data = [body_motion_data("pelvis", splines)]

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
        timestamp[],
        36,
        qstar,
        piecewise_polynomial_t(q_spline),
        17,
        [10, 17, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 36, 8, 7]
    )

    msg.num_joint_pd_overrides = 0
    msg.param_set_name = "standing"
    msg.torque_alpha_filter = 0

    LCM() do lcm
        publish(lcm, "QP_CONTROLLER_INPUT", msg)
    end
end


