OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
rz(pi/4) q[1];
cx q[0],q[5];
cx q[0],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[0],q[3];
cx q[0],q[5];
cx q[2],q[4];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[2],q[4];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
cx q[0],q[3];
rx(5*pi/4) q[0];
cx q[0],q[3];
cx q[1],q[5];
cx q[1],q[4];
cx q[1],q[3];
rx(3*pi/4) q[1];
cx q[1],q[3];
cx q[1],q[4];
cx q[1],q[5];
cx q[0],q[4];
cx q[0],q[3];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[0],q[3];
cx q[0],q[4];
cx q[5],q[0];
cx q[4],q[0];
cx q[2],q[0];
rz(7*pi/4) q[0];
cx q[2],q[0];
cx q[4],q[0];
cx q[5],q[0];
cx q[5],q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(7*pi/4) q[0];
cx q[3],q[0];
cx q[4],q[0];
cx q[5],q[0];
rz(pi/2) q[0];
rz(pi/4) q[1];
cx q[0],q[5];
cx q[0],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[0],q[3];
cx q[0],q[5];
cx q[2],q[4];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[2],q[4];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
cx q[0],q[3];
rx(5*pi/4) q[0];
cx q[0],q[3];
cx q[1],q[5];
cx q[1],q[4];
cx q[1],q[3];
rx(3*pi/4) q[1];
cx q[1],q[3];
cx q[1],q[4];
cx q[1],q[5];
cx q[0],q[4];
cx q[0],q[3];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[0],q[3];
cx q[0],q[4];
cx q[5],q[0];
cx q[4],q[0];
cx q[2],q[0];
rz(7*pi/4) q[0];
cx q[2],q[0];
cx q[4],q[0];
cx q[5],q[0];
cx q[5],q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(7*pi/4) q[0];
cx q[3],q[0];
cx q[4],q[0];
cx q[5],q[0];
rz(pi/2) q[0];
rz(pi/4) q[1];
cx q[0],q[5];
cx q[0],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[0],q[3];
cx q[0],q[5];
cx q[2],q[4];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[2],q[4];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
cx q[0],q[3];
rx(5*pi/4) q[0];
cx q[0],q[3];
cx q[1],q[5];
cx q[1],q[4];
cx q[1],q[3];
rx(3*pi/4) q[1];
cx q[1],q[3];
cx q[1],q[4];
cx q[1],q[5];
cx q[0],q[4];
cx q[0],q[3];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[0],q[3];
cx q[0],q[4];
cx q[5],q[0];
cx q[4],q[0];
cx q[2],q[0];
rz(7*pi/4) q[0];
cx q[2],q[0];
cx q[4],q[0];
cx q[5],q[0];
cx q[5],q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(7*pi/4) q[0];
cx q[3],q[0];
cx q[4],q[0];
cx q[5],q[0];
rz(pi/2) q[0];
rz(pi/4) q[1];
cx q[0],q[5];
cx q[0],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[0],q[3];
cx q[0],q[5];
cx q[2],q[4];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[2],q[4];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
cx q[0],q[3];
rx(5*pi/4) q[0];
cx q[0],q[3];
cx q[1],q[5];
cx q[1],q[4];
cx q[1],q[3];
rx(3*pi/4) q[1];
cx q[1],q[3];
cx q[1],q[4];
cx q[1],q[5];
cx q[0],q[4];
cx q[0],q[3];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[0],q[3];
cx q[0],q[4];
cx q[5],q[0];
cx q[4],q[0];
cx q[2],q[0];
rz(7*pi/4) q[0];
cx q[2],q[0];
cx q[4],q[0];
cx q[5],q[0];
cx q[5],q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(7*pi/4) q[0];
cx q[3],q[0];
cx q[4],q[0];
cx q[5],q[0];
rz(pi/2) q[0];
rz(pi/4) q[1];
cx q[0],q[5];
cx q[0],q[3];
cx q[0],q[2];
rx(pi/4) q[0];
cx q[0],q[2];
cx q[0],q[3];
cx q[0],q[5];
cx q[2],q[4];
cx q[2],q[3];
rx(pi/4) q[2];
cx q[2],q[3];
cx q[2],q[4];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
cx q[0],q[3];
rx(5*pi/4) q[0];
cx q[0],q[3];
cx q[1],q[5];
cx q[1],q[4];
cx q[1],q[3];
rx(3*pi/4) q[1];
cx q[1],q[3];
cx q[1],q[4];
cx q[1],q[5];
cx q[0],q[4];
cx q[0],q[3];
cx q[0],q[1];
rx(5*pi/4) q[0];
cx q[0],q[1];
cx q[0],q[3];
cx q[0],q[4];
cx q[5],q[0];
cx q[4],q[0];
cx q[2],q[0];
rz(7*pi/4) q[0];
cx q[2],q[0];
cx q[4],q[0];
cx q[5],q[0];
cx q[5],q[0];
cx q[4],q[0];
cx q[3],q[0];
rz(7*pi/4) q[0];
cx q[3],q[0];
cx q[4],q[0];
cx q[5],q[0];
rz(pi/2) q[0];
