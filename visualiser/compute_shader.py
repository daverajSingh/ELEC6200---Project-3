import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import math

class GaussianDominanceCompute:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create and compile compute shader
        with open('gaussian_compute.glsl', 'r') as f:
            compute_shader_source = f.read()
        
        self.compute_shader = shaders.compileShader(compute_shader_source, GL_COMPUTE_SHADER)
        self.compute_program = shaders.compileProgram(self.compute_shader)
        
        # Create output buffer
        self.output_buffer = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.output_buffer)
        # Initialize with -1 (no dominant gaussian)
        output_data = np.full(screen_width * screen_height, -1, dtype=np.int32)
        glBufferData(GL_SHADER_STORAGE_BUFFER, output_data.nbytes, output_data, GL_DYNAMIC_DRAW)
        
        # Get uniform locations
        self.uniforms = {
            'view_matrix': glGetUniformLocation(self.compute_program, 'view_matrix'),
            'projection_matrix': glGetUniformLocation(self.compute_program, 'projection_matrix'),
            'hfovxy_focal': glGetUniformLocation(self.compute_program, 'hfovxy_focal'),
            'cam_pos': glGetUniformLocation(self.compute_program, 'cam_pos'),
            'sh_dim': glGetUniformLocation(self.compute_program, 'sh_dim'),
            'scale_modifier': glGetUniformLocation(self.compute_program, 'scale_modifier'),
            'screen_size': glGetUniformLocation(self.compute_program, 'screen_size')
        }

    def dispatch(self, view_matrix, projection_matrix, hfovxy_focal, cam_pos, sh_dim, scale_modifier):
        # Use compute program
        glUseProgram(self.compute_program)
        
        # Set uniforms
        glUniformMatrix4fv(self.uniforms['view_matrix'], 1, GL_FALSE, view_matrix)
        glUniformMatrix4fv(self.uniforms['projection_matrix'], 1, GL_FALSE, projection_matrix)
        glUniform3fv(self.uniforms['hfovxy_focal'], 1, hfovxy_focal)
        glUniform3fv(self.uniforms['cam_pos'], 1, cam_pos)
        glUniform1i(self.uniforms['sh_dim'], sh_dim)
        glUniform1f(self.uniforms['scale_modifier'], scale_modifier)
        glUniform2i(self.uniforms['screen_size'], self.screen_width, self.screen_height)
        
        # Bind output buffer
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.output_buffer)
        
        # Dispatch compute shader
        num_groups_x = math.ceil(self.screen_width / 16)
        num_groups_y = math.ceil(self.screen_height / 16)
        glDispatchCompute(num_groups_x, num_groups_y, 1)
        
        # Wait for compute shader to finish
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        
    def get_result(self):
        # Read back results
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.output_buffer)
        output_data = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 
                                       self.screen_width * self.screen_height * 4)  # 4 bytes per int
        return np.frombuffer(output_data, dtype=np.int32)

    def cleanup(self):
        glDeleteProgram(self.compute_program)
        glDeleteBuffers(1, [self.output_buffer])

# # Example usage:
# # Initialize
# compute = GaussianDominanceCompute(100, 100)

# # During render loop
# compute.dispatch(view_matrix, projection_matrix, hfovxy_focal, 
#                 cam_pos, sh_dim, scale_modifier)

# # Get results when needed
# dominant_gaussians = compute.get_result()
# dominant_gaussians = dominant_gaussians.reshape(window_height, window_width)

# # Cleanup when done
# compute.cleanup()
