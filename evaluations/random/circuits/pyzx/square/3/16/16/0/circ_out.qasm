OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[3];
cx q[4],q[8];
rz(pi) q[7];
rz(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[3];
cx q[8],q[4];
rz(pi/2) q[7];
rx(pi/2) q[9];
rx(pi/2) q[11];
rx(pi/2) q[12];
rx(pi/2) q[14];
rx(pi/2) q[15];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[3];
cx q[4],q[8];
rx(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[12];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[3];
rz(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[12];
rz(pi) q[15];
rx(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[3];
rz(7*pi/4) q[7];
rx(pi/2) q[9];
rx(pi/2) q[11];
rx(pi/2) q[12];
rz(pi/2) q[15];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[3];
rz(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[12];
rx(pi/2) q[15];
rz(pi/2) q[0];
cx q[2],q[11];
rx(pi/2) q[7];
cx q[14],q[9];
rz(pi/2) q[15];
rx(pi/2) q[0];
rz(pi/2) q[7];
rz(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[14];
rz(pi/2) q[0];
rx(pi/2) q[9];
rx(pi/2) q[11];
rx(pi/2) q[14];
cx q[8],q[0];
rz(pi/2) q[9];
rz(pi/2) q[11];
rz(pi/2) q[14];
rz(pi/2) q[0];
cx q[1],q[14];
rz(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[11];
rx(pi/2) q[0];
rx(pi/2) q[8];
rx(pi/2) q[9];
rx(pi/2) q[11];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[11];
rx(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[14];
rx(pi/2) q[0];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[14];
cx q[2],q[0];
rx(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[14];
rx(pi/2) q[0];
cx q[14],q[9];
rz(pi/2) q[0];
rz(pi/2) q[9];
rz(pi/2) q[14];
cx q[0],q[8];
rx(pi/2) q[9];
rx(pi/2) q[14];
cx q[8],q[0];
rz(pi/2) q[9];
rz(pi/2) q[14];
cx q[0],q[8];
rz(pi/2) q[9];
rz(pi/2) q[0];
rz(pi/2) q[8];
rx(pi/2) q[9];
rx(pi/2) q[0];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[0];
cx q[6],q[9];
rz(pi/2) q[8];
cx q[2],q[0];
cx q[4],q[8];
rz(pi) q[6];
rz(pi/2) q[9];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[6];
rz(pi/2) q[8];
rx(pi/2) q[9];
rx(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[6];
rx(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[6];
rz(pi/2) q[8];
rz(pi/2) q[9];
cx q[0],q[8];
rz(3*pi/4) q[6];
rx(pi/2) q[9];
cx q[8],q[0];
rz(pi/2) q[6];
rz(pi/2) q[9];
cx q[0],q[8];
rx(pi/2) q[6];
rz(pi/2) q[0];
rz(pi/2) q[6];
rz(pi/2) q[8];
rx(pi/2) q[0];
cx q[6],q[9];
rx(pi/2) q[8];
rz(pi/2) q[0];
cx q[6],q[12];
rz(pi/2) q[8];
rz(pi/2) q[9];
cx q[4],q[8];
cx q[12],q[6];
rx(pi/2) q[9];
rz(pi/2) q[4];
cx q[6],q[12];
rz(pi/2) q[8];
rz(pi/2) q[9];
rx(pi/2) q[4];
rx(pi/2) q[8];
rz(pi/2) q[4];
rz(pi/2) q[8];
cx q[8],q[0];
rz(pi/2) q[0];
rz(pi/2) q[8];
rx(pi/2) q[0];
rx(pi/2) q[8];
rz(pi/2) q[0];
rz(pi/2) q[8];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[2],q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[2],q[0];
cx q[0],q[2];
cx q[2],q[0];
cx q[0],q[8];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[8];
rx(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[8];
rz(pi/2) q[0];
rz(pi/2) q[8];
rz(pi/2) q[8];
rx(pi/2) q[8];
rz(pi/2) q[8];
cx q[4],q[8];
cx q[4],q[1];
rz(pi/2) q[8];
cx q[1],q[4];
rx(pi/2) q[8];
cx q[4],q[1];
rz(pi/2) q[8];
cx q[1],q[0];
rz(pi/2) q[4];
rz(3*pi/4) q[8];
rz(pi/2) q[0];
cx q[1],q[14];
rx(pi/2) q[4];
rz(pi/2) q[8];
rx(pi/2) q[0];
rz(pi/2) q[4];
rx(pi/2) q[8];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[8];
rx(pi/2) q[14];
rz(7*pi/4) q[0];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(3*pi/4) q[14];
rx(pi/2) q[0];
rz(pi/2) q[14];
rz(pi/2) q[0];
rx(pi/2) q[14];
cx q[0],q[8];
rz(pi/2) q[14];
cx q[8],q[0];
rz(pi/2) q[14];
cx q[0],q[8];
rx(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[14];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[1],q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[0],q[2];
cx q[0],q[8];
rz(pi/2) q[2];
cx q[8],q[0];
rx(pi/2) q[2];
cx q[0],q[8];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[14];
rx(pi/2) q[0];
rz(pi/2) q[14];
rz(pi/2) q[0];
rx(pi/2) q[14];
cx q[0],q[3];
rz(pi/2) q[14];
cx q[3],q[0];
cx q[0],q[3];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[4];
rx(pi/2) q[0];
rz(5*pi/4) q[1];
rz(pi/2) q[4];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[4];
cx q[0],q[3];
rx(pi/2) q[1];
rz(pi/2) q[4];
cx q[3],q[0];
rz(pi/2) q[1];
cx q[0],q[3];
rz(5*pi/4) q[1];
rz(pi/2) q[1];
rx(pi/2) q[1];
rz(pi/2) q[1];
cx q[1],q[0];
cx q[2],q[0];
cx q[1],q[14];
cx q[0],q[2];
rz(3*pi/4) q[14];
cx q[2],q[0];
rz(pi/2) q[14];
rz(pi/2) q[0];
rx(pi/2) q[14];
rx(pi/2) q[0];
rz(pi/2) q[14];
rz(pi/2) q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[14];
rx(pi/2) q[0];
cx q[14],q[1];
rz(pi/2) q[0];
cx q[1],q[14];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[0];
cx q[0],q[2];
rz(pi/2) q[0];
cx q[6],q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[2];
cx q[14],q[2];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
rz(7*pi/4) q[2];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
cx q[2],q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(3*pi/4) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[0],q[8];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[2],q[0];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
cx q[14],q[2];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
rz(3*pi/4) q[2];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
cx q[14],q[2];
cx q[14],q[1];
rz(pi/2) q[2];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/4) q[14];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[14];
rz(pi/2) q[1];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(5*pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cx q[14],q[1];
rz(3*pi/4) q[1];
cx q[14],q[2];
rz(pi/2) q[1];
cx q[2],q[14];
rx(pi/2) q[1];
cx q[14],q[2];
cx q[2],q[0];
rz(pi/2) q[1];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[14];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[1];
cx q[2],q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(7*pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cx q[14],q[2];
cx q[2],q[14];
cx q[14],q[2];
cx q[2],q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(7*pi/4) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[0],q[8];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[2],q[0];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
cx q[14],q[2];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
rz(3*pi/4) q[2];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
cx q[14],q[2];
cx q[14],q[1];
rz(pi/2) q[2];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(5*pi/4) q[14];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[14];
rz(pi/2) q[1];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(5*pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cx q[14],q[1];
rz(3*pi/4) q[1];
cx q[14],q[2];
rz(pi/2) q[1];
cx q[2],q[14];
rx(pi/2) q[1];
cx q[14],q[2];
cx q[2],q[0];
rz(pi/2) q[1];
rz(pi/2) q[14];
cx q[1],q[0];
rx(pi/2) q[14];
cx q[0],q[1];
rz(pi/2) q[14];
cx q[1],q[0];
cx q[2],q[14];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[14];
rx(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[14];
rz(7*pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cx q[14],q[1];
rz(pi/2) q[1];
rx(pi/2) q[1];
rz(pi/2) q[1];
rz(7*pi/4) q[1];
rz(pi/2) q[1];
rx(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rx(pi/2) q[1];
rz(pi/2) q[1];
cx q[14],q[1];
rz(pi/2) q[1];
rz(pi/2) q[14];
rx(pi/2) q[1];
rx(pi/2) q[14];
rz(pi/2) q[1];
rz(pi/2) q[14];
cx q[2],q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(3*pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cx q[2],q[14];
cx q[2],q[0];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(5*pi/4) q[2];
rx(pi/2) q[14];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[14];
rz(pi/2) q[0];
cx q[1],q[14];
rx(pi/2) q[2];
cx q[14],q[1];
rz(pi/2) q[2];
cx q[1],q[14];
rz(5*pi/4) q[2];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[2];
cx q[2],q[14];
cx q[2],q[0];
rz(pi/2) q[14];
cx q[0],q[8];
rx(pi/2) q[14];
cx q[8],q[0];
rz(pi/2) q[14];
cx q[0],q[8];
cx q[2],q[0];
rz(pi/2) q[8];
cx q[1],q[0];
rx(pi/2) q[8];
cx q[0],q[1];
rz(pi/2) q[8];
cx q[1],q[0];
cx q[4],q[8];
rz(pi/2) q[0];
rz(pi/2) q[4];
rz(pi/2) q[8];
rx(pi/2) q[0];
rx(pi/2) q[4];
rx(pi/2) q[8];
rz(pi/2) q[0];
rz(pi/2) q[4];
rz(pi/2) q[8];
cx q[2],q[0];
rz(pi/2) q[8];
rz(pi/2) q[0];
rx(pi/2) q[8];
rx(pi/2) q[0];
rz(pi/2) q[8];
rz(pi/2) q[0];
rz(7*pi/4) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
cx q[0],q[1];
cx q[2],q[0];
cx q[0],q[2];
cx q[2],q[0];
cx q[2],q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(3*pi/4) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rx(pi/2) q[14];
rz(pi/2) q[14];
cx q[2],q[14];
rz(pi/2) q[2];
rz(pi/2) q[14];
rx(pi/2) q[2];
rx(pi/2) q[14];
rz(pi/2) q[2];
rz(pi/2) q[14];
cx q[0],q[2];
cx q[0],q[8];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[8];
rz(pi/2) q[2];
rx(pi/2) q[8];
rz(7*pi/4) q[2];
rz(pi/2) q[8];
rz(pi/2) q[2];
rz(pi/4) q[8];
rx(pi/2) q[2];
rz(pi/2) q[8];
rz(pi/2) q[2];
rx(pi/2) q[8];
cx q[2],q[14];
rz(pi/2) q[8];
cx q[14],q[1];
rz(pi/2) q[8];
cx q[2],q[14];
rx(pi/2) q[8];
cx q[14],q[1];
rz(pi/2) q[2];
rz(pi/2) q[8];
rx(pi/2) q[2];
rz(pi/2) q[2];
cx q[0],q[2];
cx q[0],q[8];
rz(pi/2) q[2];
rx(pi/2) q[2];
rz(pi/2) q[8];
rz(pi/2) q[2];
rx(pi/2) q[8];
cx q[2],q[11];
rz(pi/2) q[8];
cx q[0],q[8];
rz(7*pi/4) q[2];
rz(pi/2) q[11];
cx q[8],q[0];
rz(pi/2) q[2];
rx(pi/2) q[11];
cx q[0],q[8];
rx(pi/2) q[2];
rz(pi/2) q[11];
cx q[1],q[0];
rz(pi/2) q[2];
cx q[8],q[4];
rz(pi/2) q[11];
cx q[0],q[1];
cx q[14],q[2];
rz(pi/2) q[4];
rz(3*pi/4) q[8];
rx(pi/2) q[11];
cx q[1],q[0];
cx q[2],q[14];
rx(pi/2) q[4];
rz(pi/2) q[8];
rz(pi/2) q[11];
cx q[14],q[2];
rz(pi/2) q[4];
rx(pi/2) q[8];
cx q[2],q[11];
rz(pi/2) q[8];
cx q[8],q[0];
rz(pi/2) q[2];
rz(pi/2) q[11];
rz(pi/2) q[0];
rx(pi/2) q[2];
rz(5*pi/4) q[8];
rx(pi/2) q[11];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[8];
rz(pi/2) q[11];
rz(pi/2) q[0];
cx q[14],q[2];
rx(pi/2) q[8];
rz(pi/2) q[11];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[8];
cx q[14],q[9];
rx(pi/2) q[11];
rx(pi/2) q[0];
rx(pi/2) q[2];
cx q[9],q[14];
rz(pi/2) q[11];
rz(pi/2) q[0];
rz(pi/2) q[2];
cx q[14],q[9];
cx q[6],q[2];
rz(pi/2) q[14];
cx q[2],q[6];
rx(pi/2) q[14];
cx q[6],q[2];
rz(pi/2) q[14];
cx q[1],q[14];
cx q[2],q[11];
cx q[9],q[6];
cx q[1],q[0];
cx q[6],q[9];
rz(pi/2) q[11];
rz(pi/2) q[14];
rz(pi/2) q[0];
cx q[4],q[1];
cx q[9],q[6];
rx(pi/2) q[11];
rx(pi/2) q[14];
rx(pi/2) q[0];
cx q[1],q[4];
rz(pi/2) q[6];
rz(pi/2) q[11];
rz(pi/2) q[14];
rz(pi/2) q[0];
cx q[4],q[1];
rx(pi/2) q[6];
rz(pi/2) q[11];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[4];
rz(pi/2) q[6];
rx(pi/2) q[11];
rx(pi/2) q[14];
rx(pi/2) q[0];
rx(pi/2) q[4];
rz(pi/2) q[11];
rz(pi/2) q[14];
rz(pi/2) q[0];
cx q[1],q[14];
rz(pi/2) q[4];
cx q[3],q[0];
cx q[8],q[4];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[4];
rx(pi/2) q[14];
rx(pi/2) q[0];
rx(pi/2) q[3];
rx(pi/2) q[4];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[4];
rz(3*pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[14];
rx(pi/2) q[0];
rx(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[14];
cx q[2],q[0];
rz(pi/2) q[0];
cx q[2],q[6];
rx(pi/2) q[0];
rz(pi/2) q[6];
rz(pi/2) q[0];
rx(pi/2) q[6];
rz(pi/2) q[0];
rz(pi/2) q[6];
rx(pi/2) q[0];
rz(pi/2) q[6];
rz(pi/2) q[0];
rx(pi/2) q[6];
cx q[0],q[8];
rz(pi/2) q[6];
cx q[8],q[0];
cx q[0],q[8];
cx q[0],q[3];
rz(pi/2) q[0];
rz(pi/2) q[3];
rx(pi/2) q[0];
rx(pi/2) q[3];
rz(pi/2) q[0];
rz(pi/2) q[3];
