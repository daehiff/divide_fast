OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(pi) q[3];
rx(pi) q[2];
rx(pi) q[6];
cx q[4],q[8];
cx q[4],q[6];
rx(3*pi/4) q[4];
cx q[4],q[6];
cx q[4],q[8];
rz(pi/2) q[5];
rx(pi/4) q[7];
cx q[4],q[2];
cx q[0],q[4];
rx(3*pi/2) q[0];
cx q[0],q[4];
cx q[4],q[2];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
rx(7*pi/4) q[8];
cx q[1],q[3];
rx(5*pi/4) q[1];
cx q[1],q[3];
rz(7*pi/4) q[6];
cx q[6],q[2];
rz(5*pi/4) q[2];
cx q[6],q[2];
cx q[7],q[2];
cx q[0],q[7];
rx(3*pi/4) q[0];
cx q[0],q[7];
cx q[7],q[2];
cx q[8],q[4];
cx q[4],q[0];
rz(3*pi/2) q[0];
cx q[4],q[0];
cx q[8],q[4];
rz(pi/4) q[4];
cx q[6],q[5];
cx q[1],q[6];
rx(3*pi/2) q[1];
cx q[1],q[6];
cx q[6],q[5];
rz(pi) q[3];
rx(pi) q[2];
rx(pi) q[6];
cx q[4],q[8];
cx q[4],q[6];
rx(3*pi/4) q[4];
cx q[4],q[6];
cx q[4],q[8];
rz(pi/2) q[5];
rx(pi/4) q[7];
cx q[4],q[2];
cx q[0],q[4];
rx(3*pi/2) q[0];
cx q[0],q[4];
cx q[4],q[2];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
rx(7*pi/4) q[8];
cx q[1],q[3];
rx(5*pi/4) q[1];
cx q[1],q[3];
rz(7*pi/4) q[6];
cx q[6],q[2];
rz(5*pi/4) q[2];
cx q[6],q[2];
cx q[7],q[2];
cx q[0],q[7];
rx(3*pi/4) q[0];
cx q[0],q[7];
cx q[7],q[2];
cx q[8],q[4];
cx q[4],q[0];
rz(3*pi/2) q[0];
cx q[4],q[0];
cx q[8],q[4];
rz(pi/4) q[4];
cx q[6],q[5];
cx q[1],q[6];
rx(3*pi/2) q[1];
cx q[1],q[6];
cx q[6],q[5];
rz(pi) q[3];
rx(pi) q[2];
rx(pi) q[6];
cx q[4],q[8];
cx q[4],q[6];
rx(3*pi/4) q[4];
cx q[4],q[6];
cx q[4],q[8];
rz(pi/2) q[5];
rx(pi/4) q[7];
cx q[4],q[2];
cx q[0],q[4];
rx(3*pi/2) q[0];
cx q[0],q[4];
cx q[4],q[2];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
rx(7*pi/4) q[8];
cx q[1],q[3];
rx(5*pi/4) q[1];
cx q[1],q[3];
rz(7*pi/4) q[6];
cx q[6],q[2];
rz(5*pi/4) q[2];
cx q[6],q[2];
cx q[7],q[2];
cx q[0],q[7];
rx(3*pi/4) q[0];
cx q[0],q[7];
cx q[7],q[2];
cx q[8],q[4];
cx q[4],q[0];
rz(3*pi/2) q[0];
cx q[4],q[0];
cx q[8],q[4];
rz(pi/4) q[4];
cx q[6],q[5];
cx q[1],q[6];
rx(3*pi/2) q[1];
cx q[1],q[6];
cx q[6],q[5];
rz(pi) q[3];
rx(pi) q[2];
rx(pi) q[6];
cx q[4],q[8];
cx q[4],q[6];
rx(3*pi/4) q[4];
cx q[4],q[6];
cx q[4],q[8];
rz(pi/2) q[5];
rx(pi/4) q[7];
cx q[4],q[2];
cx q[0],q[4];
rx(3*pi/2) q[0];
cx q[0],q[4];
cx q[4],q[2];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
rx(7*pi/4) q[8];
cx q[1],q[3];
rx(5*pi/4) q[1];
cx q[1],q[3];
rz(7*pi/4) q[6];
cx q[6],q[2];
rz(5*pi/4) q[2];
cx q[6],q[2];
cx q[7],q[2];
cx q[0],q[7];
rx(3*pi/4) q[0];
cx q[0],q[7];
cx q[7],q[2];
cx q[8],q[4];
cx q[4],q[0];
rz(3*pi/2) q[0];
cx q[4],q[0];
cx q[8],q[4];
rz(pi/4) q[4];
cx q[6],q[5];
cx q[1],q[6];
rx(3*pi/2) q[1];
cx q[1],q[6];
cx q[6],q[5];
rz(pi) q[3];
rx(pi) q[2];
rx(pi) q[6];
cx q[4],q[8];
cx q[4],q[6];
rx(3*pi/4) q[4];
cx q[4],q[6];
cx q[4],q[8];
rz(pi/2) q[5];
rx(pi/4) q[7];
cx q[4],q[2];
cx q[0],q[4];
rx(3*pi/2) q[0];
cx q[0],q[4];
cx q[4],q[2];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
rx(7*pi/4) q[8];
cx q[1],q[3];
rx(5*pi/4) q[1];
cx q[1],q[3];
rz(7*pi/4) q[6];
cx q[6],q[2];
rz(5*pi/4) q[2];
cx q[6],q[2];
cx q[7],q[2];
cx q[0],q[7];
rx(3*pi/4) q[0];
cx q[0],q[7];
cx q[7],q[2];
cx q[8],q[4];
cx q[4],q[0];
rz(3*pi/2) q[0];
cx q[4],q[0];
cx q[8],q[4];
rz(pi/4) q[4];
cx q[6],q[5];
cx q[1],q[6];
rx(3*pi/2) q[1];
cx q[1],q[6];
cx q[6],q[5];
