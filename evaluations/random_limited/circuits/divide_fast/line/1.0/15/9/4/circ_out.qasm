OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(pi) q[0];
rx(pi) q[0];
rx(pi) q[1];
rx(pi) q[6];
rx(pi) q[7];
rz(3*pi/2) q[7];
rz(pi/4) q[3];
cx q[6],q[5];
rz(3*pi/2) q[5];
cx q[6],q[5];
rz(3*pi/4) q[2];
rx(3*pi/4) q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[3],q[7];
rx(7*pi/4) q[3];
cx q[3],q[7];
cx q[5],q[7];
rx(3*pi/4) q[5];
cx q[5],q[7];
rz(pi/2) q[3];
cx q[3],q[4];
rx(5*pi/4) q[3];
cx q[3],q[4];
rz(pi) q[0];
rx(pi) q[0];
rx(pi) q[1];
rx(pi) q[6];
rx(pi) q[7];
rz(3*pi/2) q[7];
rz(pi/4) q[3];
cx q[6],q[5];
rz(3*pi/2) q[5];
cx q[6],q[5];
rz(3*pi/4) q[2];
rx(3*pi/4) q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[3],q[7];
rx(7*pi/4) q[3];
cx q[3],q[7];
cx q[5],q[7];
rx(3*pi/4) q[5];
cx q[5],q[7];
rz(pi/2) q[3];
cx q[3],q[4];
rx(5*pi/4) q[3];
cx q[3],q[4];
rz(pi) q[0];
rx(pi) q[0];
rx(pi) q[1];
rx(pi) q[6];
rx(pi) q[7];
rz(3*pi/2) q[7];
rz(pi/4) q[3];
cx q[6],q[5];
rz(3*pi/2) q[5];
cx q[6],q[5];
rz(3*pi/4) q[2];
rx(3*pi/4) q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[3],q[7];
rx(7*pi/4) q[3];
cx q[3],q[7];
cx q[5],q[7];
rx(3*pi/4) q[5];
cx q[5],q[7];
rz(pi/2) q[3];
cx q[3],q[4];
rx(5*pi/4) q[3];
cx q[3],q[4];
rz(pi) q[0];
rx(pi) q[0];
rx(pi) q[1];
rx(pi) q[6];
rx(pi) q[7];
rz(3*pi/2) q[7];
rz(pi/4) q[3];
cx q[6],q[5];
rz(3*pi/2) q[5];
cx q[6],q[5];
rz(3*pi/4) q[2];
rx(3*pi/4) q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[3],q[7];
rx(7*pi/4) q[3];
cx q[3],q[7];
cx q[5],q[7];
rx(3*pi/4) q[5];
cx q[5],q[7];
rz(pi/2) q[3];
cx q[3],q[4];
rx(5*pi/4) q[3];
cx q[3],q[4];
rz(pi) q[0];
rx(pi) q[0];
rx(pi) q[1];
rx(pi) q[6];
rx(pi) q[7];
rz(3*pi/2) q[7];
rz(pi/4) q[3];
cx q[6],q[5];
rz(3*pi/2) q[5];
cx q[6],q[5];
rz(3*pi/4) q[2];
rx(3*pi/4) q[3];
cx q[3],q[0];
rz(pi/2) q[0];
cx q[3],q[0];
cx q[3],q[7];
rx(7*pi/4) q[3];
cx q[3],q[7];
cx q[5],q[7];
rx(3*pi/4) q[5];
cx q[5],q[7];
rz(pi/2) q[3];
cx q[3],q[4];
rx(5*pi/4) q[3];
cx q[3],q[4];
