OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(pi) q[4];
rz(pi) q[5];
cx q[6],q[5];
cx q[5],q[0];
rz(5*pi/4) q[0];
cx q[5],q[0];
cx q[6],q[5];
rx(pi) q[3];
cx q[4],q[3];
cx q[3],q[1];
rz(pi/2) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[4],q[1];
rz(7*pi/4) q[1];
cx q[4],q[1];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
rz(11*pi/4) q[7];
rz(pi/2) q[3];
rx(pi/4) q[6];
rz(5*pi/4) q[6];
rx(3*pi/4) q[2];
cx q[8],q[2];
rz(7*pi/4) q[2];
cx q[8],q[2];
cx q[1],q[6];
rx(pi/4) q[1];
cx q[1],q[6];
cx q[5],q[6];
rx(3*pi/4) q[5];
cx q[5],q[6];
cx q[7],q[5];
cx q[5],q[2];
rz(3*pi/2) q[2];
cx q[5],q[2];
cx q[7],q[5];
cx q[4],q[1];
rz(pi/2) q[1];
cx q[4],q[1];
cx q[6],q[1];
rz(7*pi/4) q[1];
cx q[6],q[1];
cx q[0],q[5];
rx(3*pi/2) q[0];
cx q[0],q[5];
cx q[6],q[5];
cx q[5],q[1];
rz(5*pi/4) q[1];
cx q[5],q[1];
cx q[6],q[5];
cx q[8],q[7];
cx q[7],q[0];
rz(pi/2) q[0];
cx q[7],q[0];
cx q[8],q[7];
rz(pi) q[4];
rz(pi) q[5];
cx q[6],q[5];
cx q[5],q[0];
rz(5*pi/4) q[0];
cx q[5],q[0];
cx q[6],q[5];
rx(pi) q[3];
cx q[4],q[3];
cx q[3],q[1];
rz(pi/2) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[4],q[1];
rz(7*pi/4) q[1];
cx q[4],q[1];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
rz(11*pi/4) q[7];
rz(pi/2) q[3];
rx(pi/4) q[6];
rz(5*pi/4) q[6];
rx(3*pi/4) q[2];
cx q[8],q[2];
rz(7*pi/4) q[2];
cx q[8],q[2];
cx q[1],q[6];
rx(pi/4) q[1];
cx q[1],q[6];
cx q[5],q[6];
rx(3*pi/4) q[5];
cx q[5],q[6];
cx q[7],q[5];
cx q[5],q[2];
rz(3*pi/2) q[2];
cx q[5],q[2];
cx q[7],q[5];
cx q[4],q[1];
rz(pi/2) q[1];
cx q[4],q[1];
cx q[6],q[1];
rz(7*pi/4) q[1];
cx q[6],q[1];
cx q[0],q[5];
rx(3*pi/2) q[0];
cx q[0],q[5];
cx q[6],q[5];
cx q[5],q[1];
rz(5*pi/4) q[1];
cx q[5],q[1];
cx q[6],q[5];
cx q[8],q[7];
cx q[7],q[0];
rz(pi/2) q[0];
cx q[7],q[0];
cx q[8],q[7];
rz(pi) q[4];
rz(pi) q[5];
cx q[6],q[5];
cx q[5],q[0];
rz(5*pi/4) q[0];
cx q[5],q[0];
cx q[6],q[5];
rx(pi) q[3];
cx q[4],q[3];
cx q[3],q[1];
rz(pi/2) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[4],q[1];
rz(7*pi/4) q[1];
cx q[4],q[1];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
rz(11*pi/4) q[7];
rz(pi/2) q[3];
rx(pi/4) q[6];
rz(5*pi/4) q[6];
rx(3*pi/4) q[2];
cx q[8],q[2];
rz(7*pi/4) q[2];
cx q[8],q[2];
cx q[1],q[6];
rx(pi/4) q[1];
cx q[1],q[6];
cx q[5],q[6];
rx(3*pi/4) q[5];
cx q[5],q[6];
cx q[7],q[5];
cx q[5],q[2];
rz(3*pi/2) q[2];
cx q[5],q[2];
cx q[7],q[5];
cx q[4],q[1];
rz(pi/2) q[1];
cx q[4],q[1];
cx q[6],q[1];
rz(7*pi/4) q[1];
cx q[6],q[1];
cx q[0],q[5];
rx(3*pi/2) q[0];
cx q[0],q[5];
cx q[6],q[5];
cx q[5],q[1];
rz(5*pi/4) q[1];
cx q[5],q[1];
cx q[6],q[5];
cx q[8],q[7];
cx q[7],q[0];
rz(pi/2) q[0];
cx q[7],q[0];
cx q[8],q[7];
rz(pi) q[4];
rz(pi) q[5];
cx q[6],q[5];
cx q[5],q[0];
rz(5*pi/4) q[0];
cx q[5],q[0];
cx q[6],q[5];
rx(pi) q[3];
cx q[4],q[3];
cx q[3],q[1];
rz(pi/2) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[4],q[1];
rz(7*pi/4) q[1];
cx q[4],q[1];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
rz(11*pi/4) q[7];
rz(pi/2) q[3];
rx(pi/4) q[6];
rz(5*pi/4) q[6];
rx(3*pi/4) q[2];
cx q[8],q[2];
rz(7*pi/4) q[2];
cx q[8],q[2];
cx q[1],q[6];
rx(pi/4) q[1];
cx q[1],q[6];
cx q[5],q[6];
rx(3*pi/4) q[5];
cx q[5],q[6];
cx q[7],q[5];
cx q[5],q[2];
rz(3*pi/2) q[2];
cx q[5],q[2];
cx q[7],q[5];
cx q[4],q[1];
rz(pi/2) q[1];
cx q[4],q[1];
cx q[6],q[1];
rz(7*pi/4) q[1];
cx q[6],q[1];
cx q[0],q[5];
rx(3*pi/2) q[0];
cx q[0],q[5];
cx q[6],q[5];
cx q[5],q[1];
rz(5*pi/4) q[1];
cx q[5],q[1];
cx q[6],q[5];
cx q[8],q[7];
cx q[7],q[0];
rz(pi/2) q[0];
cx q[7],q[0];
cx q[8],q[7];
rz(pi) q[4];
rz(pi) q[5];
cx q[6],q[5];
cx q[5],q[0];
rz(5*pi/4) q[0];
cx q[5],q[0];
cx q[6],q[5];
rx(pi) q[3];
cx q[4],q[3];
cx q[3],q[1];
rz(pi/2) q[1];
cx q[3],q[1];
cx q[4],q[3];
cx q[4],q[1];
rz(7*pi/4) q[1];
cx q[4],q[1];
cx q[7],q[2];
rz(pi/2) q[2];
cx q[7],q[2];
rz(11*pi/4) q[7];
rz(pi/2) q[3];
rx(pi/4) q[6];
rz(5*pi/4) q[6];
rx(3*pi/4) q[2];
cx q[8],q[2];
rz(7*pi/4) q[2];
cx q[8],q[2];
cx q[1],q[6];
rx(pi/4) q[1];
cx q[1],q[6];
cx q[5],q[6];
rx(3*pi/4) q[5];
cx q[5],q[6];
cx q[7],q[5];
cx q[5],q[2];
rz(3*pi/2) q[2];
cx q[5],q[2];
cx q[7],q[5];
cx q[4],q[1];
rz(pi/2) q[1];
cx q[4],q[1];
cx q[6],q[1];
rz(7*pi/4) q[1];
cx q[6],q[1];
cx q[0],q[5];
rx(3*pi/2) q[0];
cx q[0],q[5];
cx q[6],q[5];
cx q[5],q[1];
rz(5*pi/4) q[1];
cx q[5],q[1];
cx q[6],q[5];
cx q[8],q[7];
cx q[7],q[0];
rz(pi/2) q[0];
cx q[7],q[0];
cx q[8],q[7];
