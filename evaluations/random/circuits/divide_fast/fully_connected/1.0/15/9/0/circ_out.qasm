OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7],q[6];
cx q[5],q[6];
rz(pi) q[3];
rz(pi) q[5];
rx(pi) q[0];
rx(pi) q[6];
rz(5*pi/4) q[6];
cx q[0],q[5];
rx(3*pi/2) q[0];
cx q[0],q[5];
cx q[7],q[8];
rx(pi/2) q[7];
cx q[7],q[8];
cx q[8],q[2];
rz(3*pi/2) q[2];
cx q[8],q[2];
cx q[4],q[2];
cx q[5],q[6];
cx q[4],q[8];
rz(3*pi/4) q[2];
cx q[8],q[5];
rz(5*pi/4) q[5];
cx q[8],q[5];
cx q[5],q[1];
rz(7*pi/4) q[1];
cx q[5],q[1];
rz(pi/2) q[6];
rz(pi/2) q[7];
cx q[7],q[5];
rz(3*pi/4) q[5];
cx q[7],q[5];
rz(3*pi/4) q[8];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[8];
cx q[5],q[6];
rz(pi) q[3];
rz(pi) q[5];
rx(pi) q[0];
rx(pi) q[6];
rz(5*pi/4) q[6];
cx q[0],q[5];
rx(3*pi/2) q[0];
cx q[0],q[5];
cx q[7],q[8];
rx(pi/2) q[7];
cx q[7],q[8];
cx q[8],q[2];
rz(3*pi/2) q[2];
cx q[8],q[2];
cx q[4],q[2];
cx q[5],q[6];
cx q[4],q[8];
rz(3*pi/4) q[2];
cx q[8],q[5];
rz(5*pi/4) q[5];
cx q[8],q[5];
cx q[5],q[1];
rz(7*pi/4) q[1];
cx q[5],q[1];
rz(pi/2) q[6];
rz(pi/2) q[7];
cx q[7],q[5];
rz(3*pi/4) q[5];
cx q[7],q[5];
rz(3*pi/4) q[8];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[8];
cx q[5],q[6];
rz(pi) q[3];
rz(pi) q[5];
rx(pi) q[0];
rx(pi) q[6];
rz(5*pi/4) q[6];
cx q[0],q[5];
rx(3*pi/2) q[0];
cx q[0],q[5];
cx q[7],q[8];
rx(pi/2) q[7];
cx q[7],q[8];
cx q[8],q[2];
rz(3*pi/2) q[2];
cx q[8],q[2];
cx q[4],q[2];
cx q[5],q[6];
cx q[4],q[8];
rz(3*pi/4) q[2];
cx q[8],q[5];
rz(5*pi/4) q[5];
cx q[8],q[5];
cx q[5],q[1];
rz(7*pi/4) q[1];
cx q[5],q[1];
rz(pi/2) q[6];
rz(pi/2) q[7];
cx q[7],q[5];
rz(3*pi/4) q[5];
cx q[7],q[5];
rz(3*pi/4) q[8];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[8];
cx q[5],q[6];
rz(pi) q[3];
rz(pi) q[5];
rx(pi) q[0];
rx(pi) q[6];
rz(5*pi/4) q[6];
cx q[0],q[5];
rx(3*pi/2) q[0];
cx q[0],q[5];
cx q[7],q[8];
rx(pi/2) q[7];
cx q[7],q[8];
cx q[8],q[2];
rz(3*pi/2) q[2];
cx q[8],q[2];
cx q[4],q[2];
cx q[5],q[6];
cx q[4],q[8];
rz(3*pi/4) q[2];
cx q[8],q[5];
rz(5*pi/4) q[5];
cx q[8],q[5];
cx q[5],q[1];
rz(7*pi/4) q[1];
cx q[5],q[1];
rz(pi/2) q[6];
rz(pi/2) q[7];
cx q[7],q[5];
rz(3*pi/4) q[5];
cx q[7],q[5];
rz(3*pi/4) q[8];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[8];
cx q[5],q[6];
rz(pi) q[3];
rz(pi) q[5];
rx(pi) q[0];
rx(pi) q[6];
rz(5*pi/4) q[6];
cx q[0],q[5];
rx(3*pi/2) q[0];
cx q[0],q[5];
cx q[7],q[8];
rx(pi/2) q[7];
cx q[7],q[8];
cx q[8],q[2];
rz(3*pi/2) q[2];
cx q[8],q[2];
cx q[4],q[2];
cx q[5],q[6];
cx q[4],q[8];
rz(3*pi/4) q[2];
cx q[8],q[5];
rz(5*pi/4) q[5];
cx q[8],q[5];
cx q[5],q[1];
rz(7*pi/4) q[1];
cx q[5],q[1];
rz(pi/2) q[6];
rz(pi/2) q[7];
cx q[7],q[5];
rz(3*pi/4) q[5];
cx q[7],q[5];
rz(3*pi/4) q[8];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[4],q[2];
cx q[4],q[8];
cx q[7],q[6];
