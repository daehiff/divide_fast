OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0],q[7];
rx(pi/2) q[0];
cx q[0],q[7];
cx q[6],q[3];
rz(5*pi/4) q[3];
cx q[6],q[3];
rx(7*pi/4) q[2];
rx(pi/2) q[3];
cx q[4],q[0];
cx q[8],q[0];
rz(pi/2) q[0];
cx q[8],q[0];
cx q[4],q[0];
rz(pi/4) q[6];
rz(5*pi/4) q[2];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[3];
cx q[1],q[7];
rx(7*pi/4) q[1];
cx q[1],q[7];
cx q[1],q[3];
rz(pi/2) q[0];
cx q[0],q[7];
rx(pi/2) q[0];
cx q[0],q[7];
cx q[6],q[3];
rz(5*pi/4) q[3];
cx q[6],q[3];
rx(7*pi/4) q[2];
rx(pi/2) q[3];
cx q[4],q[0];
cx q[8],q[0];
rz(pi/2) q[0];
cx q[8],q[0];
cx q[4],q[0];
rz(pi/4) q[6];
rz(5*pi/4) q[2];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[3];
cx q[1],q[7];
rx(7*pi/4) q[1];
cx q[1],q[7];
cx q[1],q[3];
rz(pi/2) q[0];
cx q[0],q[7];
rx(pi/2) q[0];
cx q[0],q[7];
cx q[6],q[3];
rz(5*pi/4) q[3];
cx q[6],q[3];
rx(7*pi/4) q[2];
rx(pi/2) q[3];
cx q[4],q[0];
cx q[8],q[0];
rz(pi/2) q[0];
cx q[8],q[0];
cx q[4],q[0];
rz(pi/4) q[6];
rz(5*pi/4) q[2];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[3];
cx q[1],q[7];
rx(7*pi/4) q[1];
cx q[1],q[7];
cx q[1],q[3];
rz(pi/2) q[0];
cx q[0],q[7];
rx(pi/2) q[0];
cx q[0],q[7];
cx q[6],q[3];
rz(5*pi/4) q[3];
cx q[6],q[3];
rx(7*pi/4) q[2];
rx(pi/2) q[3];
cx q[4],q[0];
cx q[8],q[0];
rz(pi/2) q[0];
cx q[8],q[0];
cx q[4],q[0];
rz(pi/4) q[6];
rz(5*pi/4) q[2];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[3];
cx q[1],q[7];
rx(7*pi/4) q[1];
cx q[1],q[7];
cx q[1],q[3];
rz(pi/2) q[0];
cx q[0],q[7];
rx(pi/2) q[0];
cx q[0],q[7];
cx q[6],q[3];
rz(5*pi/4) q[3];
cx q[6],q[3];
rx(7*pi/4) q[2];
rx(pi/2) q[3];
cx q[4],q[0];
cx q[8],q[0];
rz(pi/2) q[0];
cx q[8],q[0];
cx q[4],q[0];
rz(pi/4) q[6];
rz(5*pi/4) q[2];
cx q[0],q[1];
rx(7*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[3];
cx q[1],q[7];
rx(7*pi/4) q[1];
cx q[1],q[7];
cx q[1],q[3];
rz(pi/2) q[0];