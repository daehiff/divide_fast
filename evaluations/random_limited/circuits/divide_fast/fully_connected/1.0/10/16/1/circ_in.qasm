OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[5],q[6];
rx(3*pi/4) q[5];
cx q[5],q[6];
cx q[1],q[3];
cx q[1],q[12];
rx(3*pi/4) q[1];
cx q[1],q[12];
cx q[1],q[3];
rx(3*pi/4) q[5];
cx q[9],q[13];
cx q[7],q[9];
rx(pi) q[7];
cx q[7],q[9];
cx q[9],q[13];
cx q[2],q[0];
cx q[8],q[0];
rz(pi) q[0];
cx q[8],q[0];
cx q[2],q[0];
rx(pi/2) q[11];
cx q[0],q[14];
cx q[0],q[2];
rx(3*pi/2) q[0];
cx q[0],q[2];
cx q[0],q[14];
cx q[11],q[0];
cx q[9],q[0];
rz(5*pi/4) q[0];
cx q[9],q[0];
cx q[11],q[0];
rz(7*pi/4) q[8];
cx q[15],q[0];
rz(pi) q[0];
cx q[15],q[0];
cx q[5],q[6];
rx(3*pi/4) q[5];
cx q[5],q[6];
cx q[1],q[3];
cx q[1],q[12];
rx(3*pi/4) q[1];
cx q[1],q[12];
cx q[1],q[3];
rx(3*pi/4) q[5];
cx q[9],q[13];
cx q[7],q[9];
rx(pi) q[7];
cx q[7],q[9];
cx q[9],q[13];
cx q[2],q[0];
cx q[8],q[0];
rz(pi) q[0];
cx q[8],q[0];
cx q[2],q[0];
rx(pi/2) q[11];
cx q[0],q[14];
cx q[0],q[2];
rx(3*pi/2) q[0];
cx q[0],q[2];
cx q[0],q[14];
cx q[11],q[0];
cx q[9],q[0];
rz(5*pi/4) q[0];
cx q[9],q[0];
cx q[11],q[0];
rz(7*pi/4) q[8];
cx q[15],q[0];
rz(pi) q[0];
cx q[15],q[0];
cx q[5],q[6];
rx(3*pi/4) q[5];
cx q[5],q[6];
cx q[1],q[3];
cx q[1],q[12];
rx(3*pi/4) q[1];
cx q[1],q[12];
cx q[1],q[3];
rx(3*pi/4) q[5];
cx q[9],q[13];
cx q[7],q[9];
rx(pi) q[7];
cx q[7],q[9];
cx q[9],q[13];
cx q[2],q[0];
cx q[8],q[0];
rz(pi) q[0];
cx q[8],q[0];
cx q[2],q[0];
rx(pi/2) q[11];
cx q[0],q[14];
cx q[0],q[2];
rx(3*pi/2) q[0];
cx q[0],q[2];
cx q[0],q[14];
cx q[11],q[0];
cx q[9],q[0];
rz(5*pi/4) q[0];
cx q[9],q[0];
cx q[11],q[0];
rz(7*pi/4) q[8];
cx q[15],q[0];
rz(pi) q[0];
cx q[15],q[0];
cx q[5],q[6];
rx(3*pi/4) q[5];
cx q[5],q[6];
cx q[1],q[3];
cx q[1],q[12];
rx(3*pi/4) q[1];
cx q[1],q[12];
cx q[1],q[3];
rx(3*pi/4) q[5];
cx q[9],q[13];
cx q[7],q[9];
rx(pi) q[7];
cx q[7],q[9];
cx q[9],q[13];
cx q[2],q[0];
cx q[8],q[0];
rz(pi) q[0];
cx q[8],q[0];
cx q[2],q[0];
rx(pi/2) q[11];
cx q[0],q[14];
cx q[0],q[2];
rx(3*pi/2) q[0];
cx q[0],q[2];
cx q[0],q[14];
cx q[11],q[0];
cx q[9],q[0];
rz(5*pi/4) q[0];
cx q[9],q[0];
cx q[11],q[0];
rz(7*pi/4) q[8];
cx q[15],q[0];
rz(pi) q[0];
cx q[15],q[0];
cx q[5],q[6];
rx(3*pi/4) q[5];
cx q[5],q[6];
cx q[1],q[3];
cx q[1],q[12];
rx(3*pi/4) q[1];
cx q[1],q[12];
cx q[1],q[3];
rx(3*pi/4) q[5];
cx q[9],q[13];
cx q[7],q[9];
rx(pi) q[7];
cx q[7],q[9];
cx q[9],q[13];
cx q[2],q[0];
cx q[8],q[0];
rz(pi) q[0];
cx q[8],q[0];
cx q[2],q[0];
rx(pi/2) q[11];
cx q[0],q[14];
cx q[0],q[2];
rx(3*pi/2) q[0];
cx q[0],q[2];
cx q[0],q[14];
cx q[11],q[0];
cx q[9],q[0];
rz(5*pi/4) q[0];
cx q[9],q[0];
cx q[11],q[0];
rz(7*pi/4) q[8];
cx q[15],q[0];
rz(pi) q[0];
cx q[15],q[0];
