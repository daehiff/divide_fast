OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3],q[1];
cx q[7],q[0];
cx q[5],q[8];
cx q[1],q[8];
cx q[5],q[0];
cx q[7],q[4];
rz(pi) q[5];
rz(pi) q[7];
cx q[7],q[3];
rz(5*pi/4) q[3];
cx q[7],q[3];
rz(pi/4) q[2];
rx(pi/4) q[7];
rx(3*pi/4) q[1];
rx(3*pi/2) q[5];
rx(5*pi/4) q[8];
rz(pi/4) q[1];
rz(3*pi/2) q[7];
rz(pi) q[5];
rz(pi) q[7];
cx q[7],q[3];
rz(5*pi/4) q[3];
cx q[7],q[3];
rz(pi/4) q[2];
rx(pi/4) q[7];
rx(3*pi/4) q[1];
rx(3*pi/2) q[5];
rx(5*pi/4) q[8];
rz(pi/4) q[1];
rz(3*pi/2) q[7];
rz(pi) q[5];
rz(pi) q[7];
cx q[7],q[3];
rz(5*pi/4) q[3];
cx q[7],q[3];
rz(pi/4) q[2];
rx(pi/4) q[7];
rx(3*pi/4) q[1];
rx(3*pi/2) q[5];
rx(5*pi/4) q[8];
rz(pi/4) q[1];
rz(3*pi/2) q[7];
rz(pi) q[5];
rz(pi) q[7];
cx q[7],q[3];
rz(5*pi/4) q[3];
cx q[7],q[3];
rz(pi/4) q[2];
rx(pi/4) q[7];
rx(3*pi/4) q[1];
rx(3*pi/2) q[5];
rx(5*pi/4) q[8];
rz(pi/4) q[1];
rz(3*pi/2) q[7];
rz(pi) q[5];
rz(pi) q[7];
cx q[7],q[3];
rz(5*pi/4) q[3];
cx q[7],q[3];
rz(pi/4) q[2];
rx(pi/4) q[7];
rx(3*pi/4) q[1];
rx(3*pi/2) q[5];
rx(5*pi/4) q[8];
rz(pi/4) q[1];
rz(3*pi/2) q[7];
cx q[7],q[4];
cx q[1],q[8];
cx q[5],q[0];
cx q[3],q[1];
cx q[7],q[0];
cx q[5],q[8];
