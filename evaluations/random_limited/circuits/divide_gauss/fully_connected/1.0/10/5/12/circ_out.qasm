OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(pi) q[1];
rz(pi) q[3];
rx(pi) q[0];
rx(pi) q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[1],q[4];
rz(pi/2) q[0];
rx(9*pi/4) q[2];
cx q[4],q[2];
rz(pi/4) q[2];
cx q[4],q[2];
rz(3*pi/4) q[4];
rx(3*pi/4) q[2];
rx(5*pi/4) q[3];
cx q[1],q[4];
rz(pi) q[1];
rz(pi) q[3];
rx(pi) q[0];
rx(pi) q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[1],q[4];
rz(pi/2) q[0];
rx(9*pi/4) q[2];
cx q[4],q[2];
rz(pi/4) q[2];
cx q[4],q[2];
rz(3*pi/4) q[4];
rx(3*pi/4) q[2];
rx(5*pi/4) q[3];
cx q[1],q[4];
rz(pi) q[1];
rz(pi) q[3];
rx(pi) q[0];
rx(pi) q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[1],q[4];
rz(pi/2) q[0];
rx(9*pi/4) q[2];
cx q[4],q[2];
rz(pi/4) q[2];
cx q[4],q[2];
rz(3*pi/4) q[4];
rx(3*pi/4) q[2];
rx(5*pi/4) q[3];
cx q[1],q[4];
rz(pi) q[1];
rz(pi) q[3];
rx(pi) q[0];
rx(pi) q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[1],q[4];
rz(pi/2) q[0];
rx(9*pi/4) q[2];
cx q[4],q[2];
rz(pi/4) q[2];
cx q[4],q[2];
rz(3*pi/4) q[4];
rx(3*pi/4) q[2];
rx(5*pi/4) q[3];
cx q[1],q[4];
rz(pi) q[1];
rz(pi) q[3];
rx(pi) q[0];
rx(pi) q[1];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[1],q[4];
rz(pi/2) q[0];
rx(9*pi/4) q[2];
cx q[4],q[2];
rz(pi/4) q[2];
cx q[4],q[2];
rz(3*pi/4) q[4];
rx(3*pi/4) q[2];
rx(5*pi/4) q[3];
cx q[1],q[4];