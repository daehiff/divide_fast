OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[12],q[0];
cx q[13],q[9];
rz(pi) q[3];
rz(pi) q[4];
rz(pi) q[6];
rx(pi) q[10];
cx q[8],q[7];
cx q[13],q[7];
rz(3*pi/4) q[7];
cx q[13],q[7];
cx q[8],q[7];
rz(5*pi/4) q[13];
rz(pi/2) q[8];
cx q[14],q[7];
rz(pi/2) q[7];
cx q[14],q[7];
cx q[1],q[9];
rx(5*pi/4) q[1];
cx q[1],q[9];
rx(pi/2) q[9];
rx(11*pi/4) q[11];
cx q[0],q[11];
cx q[0],q[15];
rx(pi/4) q[0];
cx q[0],q[15];
cx q[0],q[11];
rx(7*pi/4) q[6];
rz(pi/4) q[9];
cx q[10],q[9];
rz(pi/4) q[9];
cx q[10],q[9];
cx q[10],q[0];
rz(5*pi/4) q[0];
cx q[10],q[0];
cx q[4],q[12];
rx(7*pi/4) q[4];
cx q[4],q[12];
rz(pi) q[3];
rz(pi) q[4];
rz(pi) q[6];
rx(pi) q[10];
cx q[8],q[7];
cx q[13],q[7];
rz(3*pi/4) q[7];
cx q[13],q[7];
cx q[8],q[7];
rz(5*pi/4) q[13];
rz(pi/2) q[8];
cx q[14],q[7];
rz(pi/2) q[7];
cx q[14],q[7];
cx q[1],q[9];
rx(5*pi/4) q[1];
cx q[1],q[9];
rx(pi/2) q[9];
rx(11*pi/4) q[11];
cx q[0],q[11];
cx q[0],q[15];
rx(pi/4) q[0];
cx q[0],q[15];
cx q[0],q[11];
rx(7*pi/4) q[6];
rz(pi/4) q[9];
cx q[10],q[9];
rz(pi/4) q[9];
cx q[10],q[9];
cx q[10],q[0];
rz(5*pi/4) q[0];
cx q[10],q[0];
cx q[4],q[12];
rx(7*pi/4) q[4];
cx q[4],q[12];
rz(pi) q[3];
rz(pi) q[4];
rz(pi) q[6];
rx(pi) q[10];
cx q[8],q[7];
cx q[13],q[7];
rz(3*pi/4) q[7];
cx q[13],q[7];
cx q[8],q[7];
rz(5*pi/4) q[13];
rz(pi/2) q[8];
cx q[14],q[7];
rz(pi/2) q[7];
cx q[14],q[7];
cx q[1],q[9];
rx(5*pi/4) q[1];
cx q[1],q[9];
rx(pi/2) q[9];
rx(11*pi/4) q[11];
cx q[0],q[11];
cx q[0],q[15];
rx(pi/4) q[0];
cx q[0],q[15];
cx q[0],q[11];
rx(7*pi/4) q[6];
rz(pi/4) q[9];
cx q[10],q[9];
rz(pi/4) q[9];
cx q[10],q[9];
cx q[10],q[0];
rz(5*pi/4) q[0];
cx q[10],q[0];
cx q[4],q[12];
rx(7*pi/4) q[4];
cx q[4],q[12];
rz(pi) q[3];
rz(pi) q[4];
rz(pi) q[6];
rx(pi) q[10];
cx q[8],q[7];
cx q[13],q[7];
rz(3*pi/4) q[7];
cx q[13],q[7];
cx q[8],q[7];
rz(5*pi/4) q[13];
rz(pi/2) q[8];
cx q[14],q[7];
rz(pi/2) q[7];
cx q[14],q[7];
cx q[1],q[9];
rx(5*pi/4) q[1];
cx q[1],q[9];
rx(pi/2) q[9];
rx(11*pi/4) q[11];
cx q[0],q[11];
cx q[0],q[15];
rx(pi/4) q[0];
cx q[0],q[15];
cx q[0],q[11];
rx(7*pi/4) q[6];
rz(pi/4) q[9];
cx q[10],q[9];
rz(pi/4) q[9];
cx q[10],q[9];
cx q[10],q[0];
rz(5*pi/4) q[0];
cx q[10],q[0];
cx q[4],q[12];
rx(7*pi/4) q[4];
cx q[4],q[12];
rz(pi) q[3];
rz(pi) q[4];
rz(pi) q[6];
rx(pi) q[10];
cx q[8],q[7];
cx q[13],q[7];
rz(3*pi/4) q[7];
cx q[13],q[7];
cx q[8],q[7];
rz(5*pi/4) q[13];
rz(pi/2) q[8];
cx q[14],q[7];
rz(pi/2) q[7];
cx q[14],q[7];
cx q[1],q[9];
rx(5*pi/4) q[1];
cx q[1],q[9];
rx(pi/2) q[9];
rx(11*pi/4) q[11];
cx q[0],q[11];
cx q[0],q[15];
rx(pi/4) q[0];
cx q[0],q[15];
cx q[0],q[11];
rx(7*pi/4) q[6];
rz(pi/4) q[9];
cx q[10],q[9];
rz(pi/4) q[9];
cx q[10],q[9];
cx q[10],q[0];
rz(5*pi/4) q[0];
cx q[10],q[0];
cx q[4],q[12];
rx(7*pi/4) q[4];
cx q[4],q[12];
cx q[12],q[0];
cx q[13],q[9];
