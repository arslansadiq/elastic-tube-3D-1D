from time import time
import numpy as np
from output import writeOutputToVTK
import precice
# from thetaScheme import perform_monolithic_theta_scheme_step

# physical properties of the tube
E = 100000  # elasticity module
r0 = 1/np.sqrt(np.pi)  # radius of the tube
a0 = r0**2 * np.pi  # cross sectional area
c_mk = np.sqrt(E/2/r0)  # wave speed
p0 = 0  # pressure at outlet
u0 = 0  # mean velocity
ampl = 3  # amplitude of varying velocity
frequency = 10  # frequency of variation
t_shift = 0  # temporal shift of variation
kappa = 1000

L = 10  # length of tube/simulation domain
N = 1000
dx = L / kappa

def crossSection0(N):
    return a0 * np.ones(N + 1)

def velocity_in(t): return u0 + ampl * np.sin(frequency *
                                              (t + t_shift) * np.pi)  # inflow velocity

def dsolve_solid(pressure):
    """
    compute derivative in time of cross section area of the tube from pressure and elasticity module
    :param pressure:
    :return: new derivative in time of cross section
    """
    cross_Section0 = crossSection0(pressure.shape[0] - 1)
    # if conf.fsi_active:  # if FSI is active, cross section changes with pressure; therefore we get a non zero derivative
    pressure0 = p0 * np.ones_like(pressure)
    cross_Section = -2 * cross_Section0 * ((pressure0 - 2.0 * c_mk ** 2) ** 2 / (pressure - 2.0 * c_mk ** 2) ** 3)
    return cross_Section
    # else:  # if FSI is not active, cross section does not depend on pressure
    #     return np.zeros_like(crossSection0)

def solve_solid(pressure):
    """
    compute cross section area of the tube from pressure and elasticity module
    :param pressure:
    :return: new cross section
    """
    cross_Section0 = crossSection0(pressure.shape[0] - 1)
    # if conf.fsi_active:  # if FSI is active, cross section changes with pressure
    pressure0 = p0 * np.ones_like(pressure)
    cross_Section = cross_Section0 * ((pressure0 - 2.0 * c_mk ** 2) ** 2 / (pressure - 2.0 * c_mk ** 2) ** 2)
    return cross_Section
    # else:  # if FSI is not active, cross section stays constant
    #     return crossSection0

def perform_monolithic_theta_scheme_step(velocity0, pressure0, crossSection0, dx, tau, velocity_in, theta=1):

    k = 0

    # initial guess for Newtons method
    pressure1 = np.copy(pressure0)
    velocity1 = np.copy(velocity0)

    N = pressure0.shape[0]-1

    alpha = 0 #pp.a0 / (pp.u0 + dx/tau)
    success = True

    while success:  # perform Newton iterations to solve nonlinear system of equations

        crossSection1 = solve_solid(pressure1)
        dcrossSection1 = dsolve_solid(pressure1)

        # compute residual
        res = np.zeros(2 * N + 2)

        for i in range(1,N):
            # Momentum
            res[i] = (velocity0[i] * crossSection0[i] - velocity1[i] * crossSection1[i]) * dx / tau

            res[i] += .25 * theta * (- crossSection1[i + 1] * velocity1[i] * velocity1[i + 1] - crossSection1[i] * velocity1[i] * velocity1[i + 1])
            res[i] += .25 * (1-theta) * (- crossSection0[i + 1] * velocity0[i] * velocity0[i + 1] - crossSection0[i] * velocity0[i] * velocity0[i + 1])

            res[i] += .25 * theta * (- crossSection1[i + 1] * velocity1[i] * velocity1[i] - crossSection1[i] * velocity1[i] * velocity1[i] + crossSection1[i] * velocity1[i - 1] * velocity1[i] + crossSection1[i - 1] * velocity1[i - 1] * velocity1[i])
            res[i] += .25 * (1-theta) * (- crossSection0[i + 1] * velocity0[i] * velocity0[i] - crossSection0[i] * velocity0[i] * velocity0[i] + crossSection0[i] * velocity0[i - 1] * velocity0[i] + crossSection0[i - 1] * velocity0[i - 1] * velocity0[i])

            res[i] += .25 * theta * (+ crossSection1[i - 1] * velocity1[i - 1] * velocity1[i - 1] + crossSection1[i] * velocity1[i - 1] * velocity1[i - 1])
            res[i] += .25 * (1-theta) * (+ crossSection0[i - 1] * velocity0[i - 1] * velocity0[i - 1] + crossSection0[i] * velocity0[i - 1] * velocity0[i - 1])

            res[i] += .25 * theta * (+ crossSection1[i - 1] * pressure1[i - 1] + crossSection1[i] * pressure1[i - 1] + crossSection1[i - 1] * pressure1[i] - crossSection1[i + 1] * pressure1[i] - crossSection1[i] * pressure1[i + 1] - crossSection1[i + 1] * pressure1[i + 1])
            res[i] += .25 * (1-theta) * (+ crossSection0[i - 1] * pressure0[i - 1] + crossSection0[i] * pressure0[i - 1] + crossSection0[i - 1] * pressure0[i] - crossSection0[i + 1] * pressure0[i] - crossSection0[i] * pressure0[i + 1] - crossSection0[i + 1] * pressure0[i + 1])

            # Continuity (we only care about values at n+1, see [2],p.737,eq.(3.16-25))
            res[i + N + 1] = (crossSection0[i] - crossSection1[i]) * dx / tau
            res[i + N + 1] += .25 * theta * (+ crossSection1[i - 1] * velocity1[i - 1] + crossSection1[i] * velocity1[i - 1] + crossSection1[i - 1] * velocity1[i] - crossSection1[i + 1] * velocity1[i] - crossSection1[i] * velocity1[i + 1] - crossSection1[i + 1] * velocity1[i + 1])
            res[i + N + 1] += .25 * (1-theta) * (+ crossSection0[i - 1] * velocity0[i - 1] + crossSection0[i] * velocity0[i - 1] + crossSection0[i - 1] * velocity0[i] - crossSection0[i + 1] * velocity0[i] - crossSection0[i] * velocity0[i + 1] - crossSection0[i + 1] * velocity0[i + 1])
            res[i + N + 1] += alpha * theta * (pressure1[i - 1] - 2 * pressure1[i] + pressure1[i + 1])

        # Boundary

        # Velocity Inlet is prescribed
        res[0] = velocity_in - velocity1[0]

        # Pressure Inlet is lineary interpolated
        res[N + 1] = -pressure1[0] + 2 * pressure1[1] - pressure1[2]

        # Velocity Outlet is lineary interpolated
        res[N] = -velocity1[-1] + 2 * velocity1[-2] - velocity1[-3]

        # if pp.fsi_active:
        # Pressure Outlet is "non-reflecting"
        tmp2 = np.sqrt(c_mk**2 - pressure0[-1] / 2) - (velocity1[-1] - velocity0[-1]) / 4
        res[2 * N + 1] = -pressure1[-1] + 2 * (c_mk**2 - tmp2 * tmp2)
        # else:
            # Pressure Outlet is prescribed
            # res[2 * N + 1] = p0 - pressure1[-1]
            #tmp2 = np.sqrt(1 - pressure0[N] / 2) - (velocity1[N] - velocity0[N]) / 4
            #res[2 * N + 1] = -pressure1[N] + 2 * (1 - tmp2 * tmp2)


        k += 1  # Iteration Count

        # compute relative norm of residual
        norm_1 = np.sqrt(res.dot(res))
        norm_2 = np.sqrt(pressure1.dot(pressure1) + velocity1.dot(velocity1))
        norm = norm_1 / norm_2

        if norm < 1e-10 and k > 1:
            break  # Nonlinear Solver success
        elif k > 1000:
            print ("Nonlinear Solver break, iterations: %i, residual norm: %e\n" % (k, norm))
            velocity1[:] = np.nan
            pressure1[:] = np.nan
            crossSection1[:] = np.nan
            success = False
            break
        # else:
        # perform another iteration of newton's method

        # compute Jacobian for Newton's method
        system = np.zeros([N+N+2,N+N+2])

        for i in range(1,N):
            ### Momentum, Velocity see [1] eq. (13b) ###

            # df[i]/du[i-1], f[i] = -res[i]
            system[i][i - 1] += 0.25 * theta * (- crossSection1[i] * velocity1[i] -  crossSection1[i - 1] * velocity1[i])
            system[i][i - 1] += 0.25 * theta * (- 2 * crossSection1[i - 1] * velocity1[i - 1] - 2 * crossSection1[i] * velocity1[i - 1])
            # df[i]/du[i], f[i] = -res[i]
            system[i][i] += crossSection1[i] * dx/tau
            system[i][i] += 0.25 * theta * (+ crossSection1[i + 1] * velocity1[i + 1] + crossSection1[i] * velocity1[i + 1])
            system[i][i] += 0.25 * theta * (+ 2 * crossSection1[i + 1] * velocity1[i] + 2 * crossSection1[i] * velocity1[i] - crossSection1[i] * velocity1[i - 1] - crossSection1[i - 1] * velocity1[i - 1])
            # df[i]/du[i+1], f[i] = -res[i]
            system[i][i + 1] += 0.25 * theta * (crossSection1[i + 1] * velocity1[i] + crossSection1[i] * velocity1[i])

            ### Momentum, Pressure see [1] eq. (13b) ###

            # df[i]/dp[i-1], f[i] = -res[i]
            system[i][N + 1 + i - 1] += 0.25 * theta * (- dcrossSection1[i - 1] * velocity1[i - 1] * velocity1[i])
            system[i][N + 1 + i - 1] += 0.25 * theta * (- dcrossSection1[i - 1] * velocity1[i - 1] * velocity1[i - 1])
            system[i][N + 1 + i - 1] += 0.25 * theta * (- crossSection1[i - 1] - dcrossSection1[i - 1] * pressure1[i - 1] - crossSection1[i] - dcrossSection1[i - 1] * pressure1[i])
            # df[i]/dp[i], f[i] = -res[i]
            system[i][N + 1 + i] += velocity1[i] * dcrossSection1[i] * dx/tau
            system[i][N + 1 + i] += 0.25 * theta * (+ dcrossSection1[i] * velocity1[i] * velocity1[i + 1])
            system[i][N + 1 + i] += 0.25 * theta * (+ dcrossSection1[i] * velocity1[i] * velocity1[i] - dcrossSection1[i] * velocity1[i - 1] * velocity1[i])
            system[i][N + 1 + i] += 0.25 * theta * (- dcrossSection1[i] * velocity1[i - 1] * velocity1[i - 1])
            system[i][N + 1 + i] += 0.25 * theta * (- crossSection1[i - 1] + crossSection1[i + 1] - dcrossSection1[i] * pressure1[i - 1] + dcrossSection1[i] * pressure1[i + 1])
            # df[i]/dp[i+1], f[i] = -res[i]
            system[i][N + 1 + i + 1] += 0.25 * theta * (+ dcrossSection1[i + 1] * velocity1[i] * velocity1[i + 1])
            system[i][N + 1 + i + 1] += 0.25 * theta * (+ dcrossSection1[i + 1] * velocity1[i] * velocity1[i])
            system[i][N + 1 + i + 1] += 0.25 * theta * (+ dcrossSection1[i + 1] * pressure1[i] + crossSection1[i] + dcrossSection1[i + 1] * pressure1[i + 1] + crossSection1[i + 1])

            ### Continuity, Velocity see [1] eq. (13a) ###

            # df[i]/du[i-1], f[i] = -res[i]
            system[i + N + 1][i - 1] += 0.25 * theta * (- crossSection1[i - 1] - crossSection1[i])
            # df[i]/du[i], f[i] = -res[i]
            system[i + N + 1][i] += 0.25 * theta * (- crossSection1[i - 1] + crossSection1[i + 1])
            # df[i]/du[i+1], f[i] = -res[i]
            system[i + N + 1][i + 1] += 0.25 * theta * (+ crossSection1[i] + crossSection1[i + 1])

            # Continuity, Pressure see [1] eq. (13a)

            # dg[i]/dp[i-1], g[i] = -res[i + N + 1]
            system[i + N + 1][N + 1 + i - 1] += 0.25 * theta * (- dcrossSection1[i - 1] * velocity1[i - 1] - dcrossSection1[i - 1] * velocity1[i])
            system[i + N + 1][N + 1 + i - 1] += - alpha * theta
            # dg[i]/dp[i], g[i] = -res[i + N + 1]
            system[i + N + 1][N + 1 + i] += dcrossSection1[i] * dx / tau
            system[i + N + 1][N + 1 + i] += 0.25 * theta * (- dcrossSection1[i] * velocity1[i - 1] + dcrossSection1[i] * velocity1[i + 1])
            system[i + N + 1][N + 1 + i] += 2 * alpha * theta
            # dg[i]/dp[i+1], g[i] = -res[i + N + 1]
            system[i + N + 1][N + 1 + i + 1] += 0.25 * theta * (+ dcrossSection1[i + 1] * velocity1[i] + dcrossSection1[i + 1] * velocity1[i + 1])
            system[i + N + 1][N + 1 + i + 1] += - alpha * theta

        # Velocity Inlet is prescribed
        system[0][0] = 1
        # Pressure Inlet is linearly interpolated [1] eq. (14a)
        system[N + 1][N + 1] = 1
        system[N + 1][N + 2] = -2
        system[N + 1][N + 3] = 1
        # Velocity Outlet is linearly interpolated [1] eq. (14b)
        system[N][N] = 1
        system[N][N - 1] = -2
        system[N][N - 2] = 1
        # if pp.fsi_active:
        # Pressure Outlet is Non-Reflecting [1] eq. (15)
        system[2 * N + 1][2 * N + 1] = 1
        system[2 * N + 1][N] = -(np.sqrt(c_mk**2 - pressure0[-1] / 2) - (velocity1[-1] - velocity0[-1]) / 4)
        # else:
        #     # Pressure Outlet is prescribed
        #     system[2 * N + 1][2 * N + 1] = 1

        try:
            solution = np.linalg.solve(system, res)
        except np.linalg.LinAlgError:
            print ("LINALGERROR! SINGULAR MATRIX")
            velocity1[:] = np.nan
            pressure1[:] = np.nan
            crossSection1[:] = np.nan
            success = False
            break

        velocity1 += solution[:N + 1]
        pressure1 += solution[N + 1:]

    return velocity1, pressure1, crossSection1, success

'''preCICE setup'''
configFileName = "../precice-config.xml"
participantName = "MonolithicTube1D"
meshName = participantName + "Inlet-Mesh"
solverProcessIndex = 0
solverProcessSize = 1
interface = precice.Interface(participantName, configFileName, solverProcessIndex, solverProcessSize)
meshID = interface.get_mesh_id(meshName)
readData = "Velocity"
readdataID = interface.get_data_id(readData, meshID)
vertex = np.array([0, 0, 0.05])
vertexID = interface.set_mesh_vertex(meshID, vertex)

'''Initializing velocities and pressure'''
velocity = velocity_in(0) * np.ones(N + 1)
velocity_old = velocity_in(0) * np.ones(N + 1)
pressure = p0 * np.ones(N + 1)
pressure_old = p0 * np.ones(N + 1)
crossSectionLength = a0 * np.ones(N + 1)
crossSectionLength_old = a0 * np.ones(N + 1)

crossSectionLength_old = np.copy(crossSectionLength)
# initialize such that mass conservation is fulfilled
# velocity_old = velocity_in(
#     0) * crossSectionLength_old[0] * np.ones(N + 1) / crossSectionLength_old

# t_end = 1
# dt = 0.001
# total_step = t_end/dt

# t = 0
time_it = 0

precice_dt = interface.initialize()

while interface.is_coupling_ongoing():
    print("in the while loop")
    readData = interface.read_vector_data(readdataID, vertexID)
    print(readData)
    v_in = readData[-1]
    velocity, pressure, crossSectionLength, success = perform_monolithic_theta_scheme_step(
        velocity_old, pressure_old, crossSectionLength_old, 
        dx, precice_dt, v_in)
    print("Done for time: ",(time_it))
    velocity_old = np.copy(velocity)
    pressure_old = np.copy(pressure)
    crossSectionLength_old = np.copy(crossSectionLength)
    
    writeOutputToVTK(time_it, "out_fluid_", dx, datanames=["velocity", "pressure", "diameter"], data=[
            velocity_old, pressure_old, crossSectionLength_old])
    time_it = time_it + 1
    interface.advance(precice_dt)

print("Exiting FluidSolver")
