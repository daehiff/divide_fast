OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rz(pi/2) q[0];
rz(pi/2) q[2];
cx q[1],q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
cx q[1],q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi) q[0];
rz(pi) q[1];
rx(pi) q[0];
rx(pi) q[1];
