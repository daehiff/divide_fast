OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[3];
rz(pi) q[1];
cx q[0],q[5];
rx(3*pi/4) q[0];
cx q[0],q[5];
cx q[3],q[6];
rx(3*pi/2) q[3];
cx q[3],q[6];
rx(pi) q[6];
rz(3*pi/2) q[5];
cx q[5],q[8];
rx(pi/4) q[5];
cx q[5],q[8];
cx q[7],q[5];
rz(7*pi/4) q[5];
cx q[7],q[5];
cx q[3],q[1];
cx q[2],q[1];
rz(pi) q[1];
cx q[2],q[1];
cx q[3],q[1];
cx q[2],q[7];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
cx q[2],q[7];
cx q[3],q[6];
cx q[3],q[7];
rx(3*pi/2) q[3];
cx q[3],q[7];
cx q[3],q[6];
cx q[2],q[7];
cx q[2],q[5];
rx(7*pi/4) q[2];
cx q[2],q[5];
cx q[2],q[7];
rx(5*pi/4) q[3];
cx q[5],q[3];
rz(5*pi/4) q[3];
cx q[5],q[3];
cx q[8],q[7];
cx q[4],q[8];
rx(pi) q[4];
cx q[4],q[8];
cx q[8],q[7];
cx q[5],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[3];
rz(pi) q[1];
cx q[0],q[5];
rx(3*pi/4) q[0];
cx q[0],q[5];
cx q[3],q[6];
rx(3*pi/2) q[3];
cx q[3],q[6];
rx(pi) q[6];
rz(3*pi/2) q[5];
cx q[5],q[8];
rx(pi/4) q[5];
cx q[5],q[8];
cx q[7],q[5];
rz(7*pi/4) q[5];
cx q[7],q[5];
cx q[3],q[1];
cx q[2],q[1];
rz(pi) q[1];
cx q[2],q[1];
cx q[3],q[1];
cx q[2],q[7];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
cx q[2],q[7];
cx q[3],q[6];
cx q[3],q[7];
rx(3*pi/2) q[3];
cx q[3],q[7];
cx q[3],q[6];
cx q[2],q[7];
cx q[2],q[5];
rx(7*pi/4) q[2];
cx q[2],q[5];
cx q[2],q[7];
rx(5*pi/4) q[3];
cx q[5],q[3];
rz(5*pi/4) q[3];
cx q[5],q[3];
cx q[8],q[7];
cx q[4],q[8];
rx(pi) q[4];
cx q[4],q[8];
cx q[8],q[7];
cx q[5],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[3];
rz(pi) q[1];
cx q[0],q[5];
rx(3*pi/4) q[0];
cx q[0],q[5];
cx q[3],q[6];
rx(3*pi/2) q[3];
cx q[3],q[6];
rx(pi) q[6];
rz(3*pi/2) q[5];
cx q[5],q[8];
rx(pi/4) q[5];
cx q[5],q[8];
cx q[7],q[5];
rz(7*pi/4) q[5];
cx q[7],q[5];
cx q[3],q[1];
cx q[2],q[1];
rz(pi) q[1];
cx q[2],q[1];
cx q[3],q[1];
cx q[2],q[7];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
cx q[2],q[7];
cx q[3],q[6];
cx q[3],q[7];
rx(3*pi/2) q[3];
cx q[3],q[7];
cx q[3],q[6];
cx q[2],q[7];
cx q[2],q[5];
rx(7*pi/4) q[2];
cx q[2],q[5];
cx q[2],q[7];
rx(5*pi/4) q[3];
cx q[5],q[3];
rz(5*pi/4) q[3];
cx q[5],q[3];
cx q[8],q[7];
cx q[4],q[8];
rx(pi) q[4];
cx q[4],q[8];
cx q[8],q[7];
cx q[5],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[3];
rz(pi) q[1];
cx q[0],q[5];
rx(3*pi/4) q[0];
cx q[0],q[5];
cx q[3],q[6];
rx(3*pi/2) q[3];
cx q[3],q[6];
rx(pi) q[6];
rz(3*pi/2) q[5];
cx q[5],q[8];
rx(pi/4) q[5];
cx q[5],q[8];
cx q[7],q[5];
rz(7*pi/4) q[5];
cx q[7],q[5];
cx q[3],q[1];
cx q[2],q[1];
rz(pi) q[1];
cx q[2],q[1];
cx q[3],q[1];
cx q[2],q[7];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
cx q[2],q[7];
cx q[3],q[6];
cx q[3],q[7];
rx(3*pi/2) q[3];
cx q[3],q[7];
cx q[3],q[6];
cx q[2],q[7];
cx q[2],q[5];
rx(7*pi/4) q[2];
cx q[2],q[5];
cx q[2],q[7];
rx(5*pi/4) q[3];
cx q[5],q[3];
rz(5*pi/4) q[3];
cx q[5],q[3];
cx q[8],q[7];
cx q[4],q[8];
rx(pi) q[4];
cx q[4],q[8];
cx q[8],q[7];
cx q[5],q[3];
cx q[4],q[3];
rz(5*pi/4) q[3];
cx q[4],q[3];
cx q[5],q[3];
rz(pi) q[1];
cx q[0],q[5];
rx(3*pi/4) q[0];
cx q[0],q[5];
cx q[3],q[6];
rx(3*pi/2) q[3];
cx q[3],q[6];
rx(pi) q[6];
rz(3*pi/2) q[5];
cx q[5],q[8];
rx(pi/4) q[5];
cx q[5],q[8];
cx q[7],q[5];
rz(7*pi/4) q[5];
cx q[7],q[5];
cx q[3],q[1];
cx q[2],q[1];
rz(pi) q[1];
cx q[2],q[1];
cx q[3],q[1];
cx q[2],q[7];
cx q[2],q[6];
rx(3*pi/2) q[2];
cx q[2],q[6];
cx q[2],q[7];
cx q[3],q[6];
cx q[3],q[7];
rx(3*pi/2) q[3];
cx q[3],q[7];
cx q[3],q[6];
cx q[2],q[7];
cx q[2],q[5];
rx(7*pi/4) q[2];
cx q[2],q[5];
cx q[2],q[7];
rx(5*pi/4) q[3];
cx q[5],q[3];
rz(5*pi/4) q[3];
cx q[5],q[3];
cx q[8],q[7];
cx q[4],q[8];
rx(pi) q[4];
cx q[4],q[8];
cx q[8],q[7];