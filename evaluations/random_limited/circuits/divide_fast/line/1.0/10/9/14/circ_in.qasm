OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7],q[2];
cx q[3],q[2];
rz(5*pi/4) q[2];
cx q[3],q[2];
cx q[7],q[2];
cx q[6],q[2];
cx q[4],q[2];
rz(5*pi/4) q[2];
cx q[4],q[2];
cx q[6],q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(5*pi/4) q[2];
cx q[3],q[2];
cx q[4],q[2];
cx q[5],q[4];
rz(pi/2) q[4];
cx q[5],q[4];
rx(pi/4) q[4];
cx q[6],q[3];
cx q[7],q[3];
rz(pi/2) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(3*pi/2) q[1];
rz(pi/2) q[2];
rz(3*pi/4) q[5];
rz(3*pi/2) q[8];
cx q[7],q[2];
cx q[3],q[2];
rz(5*pi/4) q[2];
cx q[3],q[2];
cx q[7],q[2];
cx q[6],q[2];
cx q[4],q[2];
rz(5*pi/4) q[2];
cx q[4],q[2];
cx q[6],q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(5*pi/4) q[2];
cx q[3],q[2];
cx q[4],q[2];
cx q[5],q[4];
rz(pi/2) q[4];
cx q[5],q[4];
rx(pi/4) q[4];
cx q[6],q[3];
cx q[7],q[3];
rz(pi/2) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(3*pi/2) q[1];
rz(pi/2) q[2];
rz(3*pi/4) q[5];
rz(3*pi/2) q[8];
cx q[7],q[2];
cx q[3],q[2];
rz(5*pi/4) q[2];
cx q[3],q[2];
cx q[7],q[2];
cx q[6],q[2];
cx q[4],q[2];
rz(5*pi/4) q[2];
cx q[4],q[2];
cx q[6],q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(5*pi/4) q[2];
cx q[3],q[2];
cx q[4],q[2];
cx q[5],q[4];
rz(pi/2) q[4];
cx q[5],q[4];
rx(pi/4) q[4];
cx q[6],q[3];
cx q[7],q[3];
rz(pi/2) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(3*pi/2) q[1];
rz(pi/2) q[2];
rz(3*pi/4) q[5];
rz(3*pi/2) q[8];
cx q[7],q[2];
cx q[3],q[2];
rz(5*pi/4) q[2];
cx q[3],q[2];
cx q[7],q[2];
cx q[6],q[2];
cx q[4],q[2];
rz(5*pi/4) q[2];
cx q[4],q[2];
cx q[6],q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(5*pi/4) q[2];
cx q[3],q[2];
cx q[4],q[2];
cx q[5],q[4];
rz(pi/2) q[4];
cx q[5],q[4];
rx(pi/4) q[4];
cx q[6],q[3];
cx q[7],q[3];
rz(pi/2) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(3*pi/2) q[1];
rz(pi/2) q[2];
rz(3*pi/4) q[5];
rz(3*pi/2) q[8];
cx q[7],q[2];
cx q[3],q[2];
rz(5*pi/4) q[2];
cx q[3],q[2];
cx q[7],q[2];
cx q[6],q[2];
cx q[4],q[2];
rz(5*pi/4) q[2];
cx q[4],q[2];
cx q[6],q[2];
cx q[4],q[2];
cx q[3],q[2];
rz(5*pi/4) q[2];
cx q[3],q[2];
cx q[4],q[2];
cx q[5],q[4];
rz(pi/2) q[4];
cx q[5],q[4];
rx(pi/4) q[4];
cx q[6],q[3];
cx q[7],q[3];
rz(pi/2) q[3];
cx q[7],q[3];
cx q[6],q[3];
rz(3*pi/2) q[1];
rz(pi/2) q[2];
rz(3*pi/4) q[5];
rz(3*pi/2) q[8];
