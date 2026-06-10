"""
PID Controller for temperature control
"""

import numpy as np


class PIDController:
    """
    PID (Proportional-Integral-Derivative) controller for temperature control
    """

    def __init__(self, Kp=2.0, Ki=0.1, Kd=0.05, setpoint=20.0, 
                 output_limits=(20.0, 50.0)):
        """
        Parameters
        ----------
        Kp : float
            Proportional gain
        Ki : float
            Integral gain
        Kd : float
            Derivative gain
        setpoint : float
            Temperature setpoint [°C]
        output_limits : tuple
            (min, max) output limits [°C]
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self.integral = 0.0
        self.last_error = 0.0

        # Anti-windup limits for integral term
        self.integral_limits = (-5000, 5000)

    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.last_error = 0.0

    def update(self, measured_value, dt=3600.0):
        """
        Update PID controller

        Parameters
        ----------
        measured_value : float
            Current measured temperature [°C]
        dt : float
            Time step [s]

        Returns
        -------
        output : float
            Control output (supply temperature) [°C]
        """
        # Calculate error
        error = self.setpoint - measured_value

        # Clamp error to prevent extreme values
        error = np.clip(error, -50, 50)

        # Proportional term
        P = self.Kp * error

        # Integral term with strong anti-windup
        self.integral += error * dt / 3600  # Normalize to hours
        self.integral = np.clip(self.integral, 
                               self.integral_limits[0], 
                               self.integral_limits[1])
        I = self.Ki * self.integral

        # Derivative term
        if dt > 0:
            derivative = (error - self.last_error) / (dt / 3600)
        else:
            derivative = 0.0
        derivative = np.clip(derivative, -10, 10)  # Limit derivative
        D = self.Kd * derivative

        # Update last error
        self.last_error = error

        # Calculate output
        output = self.setpoint + P + I + D

        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Anti-windup: back-calculate integral if saturated
        if output >= self.output_limits[1] and error > 0:
            self.integral *= 0.9  # Reduce integral
        elif output <= self.output_limits[0] and error < 0:
            self.integral *= 0.9  # Reduce integral

        return output
