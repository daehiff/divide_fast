OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0],q[7];
cx q[4],q[5];
cx q[1],q[2];
cx q[8],q[6];
cx q[7],q[6];
cx q[8],q[0];
cx q[0],q[7];
cx q[4],q[8];
rz(5*pi/4) q[3];
rz(3*pi/2) q[0];
cx q[7],q[0];
rz(3*pi/2) q[0];
cx q[7],q[0];
cx q[3],q[5];
rx(3*pi/2) q[3];
cx q[3],q[5];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[6],q[3];
cx q[7],q[3];
rz(3*pi/2) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(3*pi/4) q[5];
rx(3*pi/2) q[3];
cx q[5],q[6];
rx(pi/2) q[5];
cx q[5],q[6];
rz(5*pi/4) q[3];
rz(3*pi/2) q[0];
cx q[7],q[0];
rz(3*pi/2) q[0];
cx q[7],q[0];
cx q[3],q[5];
rx(3*pi/2) q[3];
cx q[3],q[5];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[6],q[3];
cx q[7],q[3];
rz(3*pi/2) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(3*pi/4) q[5];
rx(3*pi/2) q[3];
cx q[5],q[6];
rx(pi/2) q[5];
cx q[5],q[6];
rz(5*pi/4) q[3];
rz(3*pi/2) q[0];
cx q[7],q[0];
rz(3*pi/2) q[0];
cx q[7],q[0];
cx q[3],q[5];
rx(3*pi/2) q[3];
cx q[3],q[5];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[6],q[3];
cx q[7],q[3];
rz(3*pi/2) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(3*pi/4) q[5];
rx(3*pi/2) q[3];
cx q[5],q[6];
rx(pi/2) q[5];
cx q[5],q[6];
rz(5*pi/4) q[3];
rz(3*pi/2) q[0];
cx q[7],q[0];
rz(3*pi/2) q[0];
cx q[7],q[0];
cx q[3],q[5];
rx(3*pi/2) q[3];
cx q[3],q[5];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[6],q[3];
cx q[7],q[3];
rz(3*pi/2) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(3*pi/4) q[5];
rx(3*pi/2) q[3];
cx q[5],q[6];
rx(pi/2) q[5];
cx q[5],q[6];
rz(5*pi/4) q[3];
rz(3*pi/2) q[0];
cx q[7],q[0];
rz(3*pi/2) q[0];
cx q[7],q[0];
cx q[3],q[5];
rx(3*pi/2) q[3];
cx q[3],q[5];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[6],q[3];
cx q[7],q[3];
rz(3*pi/2) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(3*pi/4) q[5];
rx(3*pi/2) q[3];
cx q[5],q[6];
rx(pi/2) q[5];
cx q[5],q[6];
cx q[0],q[7];
cx q[4],q[8];
cx q[7],q[6];
cx q[8],q[0];
cx q[0],q[7];
cx q[4],q[5];
cx q[1],q[2];
cx q[8],q[6];
