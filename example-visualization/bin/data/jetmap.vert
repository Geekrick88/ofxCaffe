uniform sampler2DRect image;
uniform float maxValue;
varying vec2 texcoordM;
const float PI = 3.14159265;
const float c1 = 0.0;
const float colorMax = 1.0;

void main() {
    // get the homogeneous 2d position
    gl_Position = ftransform();
    
    // transform texcoord	
	vec2 texcoord = vec2(gl_TextureMatrix[0] * gl_MultiTexCoord0);
	
	// get sample positions
    texcoordM = texcoord;
}
