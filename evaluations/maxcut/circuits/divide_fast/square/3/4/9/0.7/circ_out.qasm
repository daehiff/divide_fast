OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rx(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
rz(pi/2) q[3];
cx q[2],q[3];
rx(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
rz(pi/2) q[3];
cx q[2],q[3];
rx(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/2) q[1];
cx q[2],q[1];
cx q[3],q[2];
rz(pi/2) q[3];
cx q[2],q[3];
