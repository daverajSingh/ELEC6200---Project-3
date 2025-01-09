#version 430

layout(local_size_x = 16, local_size_y = 16) in;

// Input buffers (same as vertex shader)
layout (std430, binding=0) buffer gaussian_data {
    float g_data[];
};
layout (std430, binding=1) buffer gaussian_order {
    int gi[];
};

// Output buffer to store the dominant Gaussian index for each pixel
layout (std430, binding=2) buffer output_buffer {
    int dominant_gaussian[];
};

// Uniforms (matching your vertex shader)
uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 hfovxy_focal;
uniform vec3 cam_pos;
uniform int sh_dim;
uniform float scale_modifier;
uniform ivec2 screen_size;

// Constants from vertex shader
#define POS_IDX 0
#define ROT_IDX 3
#define SCALE_IDX 7
#define OPACITY_IDX 10
#define SH_IDX 11

// Helper functions from your vertex shader
mat3 computeCov3D(vec3 scale, vec4 q) {
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    mat3 R = mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    mat3 M = S * R;
    return transpose(M) * M;
}

vec3 computeCov2D(vec4 mean_view, float focal_x, float focal_y, float tan_fovx, float tan_fovy, mat3 cov3D, mat4 viewmatrix) {
    vec4 t = mean_view;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    mat3 J = mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0
    );
    mat3 W = transpose(mat3(viewmatrix));
    mat3 T = W * J;

    mat3 cov = transpose(T) * transpose(cov3D) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

vec3 get_vec3(int offset) {
    return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
}

vec4 get_vec4(int offset) {
    return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
}

void main() {
    // Get current pixel coordinates
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= screen_size.x || pixel.y >= screen_size.y) {
        return;
    }

    // Convert pixel coordinates to NDC space
    vec2 ndc = (vec2(pixel) / vec2(screen_size)) * 2.0 - 1.0;
    
    float max_opacity = 0.0;
    int dominant_idx = -1;
    int total_dim = 3 + 4 + 3 + 1 + sh_dim;
    
    // Iterate through all Gaussians
    for (int i = 0; i < gi.length(); i++) {
        int boxid = gi[i];
        int start = boxid * total_dim;
        
        // Get Gaussian parameters
        vec4 g_pos = vec4(get_vec3(start + POS_IDX), 1.f);
        vec4 g_pos_view = view_matrix * g_pos;
        vec4 g_pos_screen = projection_matrix * g_pos_view;
        g_pos_screen.xyz = g_pos_screen.xyz / g_pos_screen.w;
        
        // Early culling
        if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3)))) {
            continue;
        }
        
        vec4 g_rot = get_vec4(start + ROT_IDX);
        vec3 g_scale = get_vec3(start + SCALE_IDX);
        float g_opacity = g_data[start + OPACITY_IDX];
        
        // Compute covariance matrices
        mat3 cov3d = computeCov3D(g_scale * scale_modifier, g_rot);
        vec2 wh = 2 * hfovxy_focal.xy * hfovxy_focal.z;
        vec3 cov2d = computeCov2D(g_pos_view, 
                                 hfovxy_focal.z, 
                                 hfovxy_focal.z, 
                                 hfovxy_focal.x, 
                                 hfovxy_focal.y, 
                                 cov3d, 
                                 view_matrix);
        
        // Calculate determinant
        float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
        if (det == 0.0f) {
            continue;
        }
        
        // Calculate conic parameters
        float det_inv = 1.f / det;
        vec3 conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
        
        // Convert screen position to local coordinates
        vec2 gaussian_ndc = g_pos_screen.xy;
        vec2 diff = (ndc - gaussian_ndc) * wh / 2.0;
        
        // Calculate power and opacity
        float power = -0.5f * (conic.x * diff.x * diff.x + conic.z * diff.y * diff.y) - conic.y * diff.x * diff.y;
        if (power <= 0.f) {
            float opacity = min(0.99f, g_opacity * exp(power));
            
            if (opacity > max_opacity && opacity >= 1.f/255.f) {
                max_opacity = opacity;
                dominant_idx = boxid;
            }
        }
    }
    
    // Store the result
    dominant_gaussian[pixel.y * screen_size.x + pixel.x] = dominant_idx;
}