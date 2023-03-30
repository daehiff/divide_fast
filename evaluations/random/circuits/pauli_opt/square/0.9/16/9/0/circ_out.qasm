OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3],q[8];
cx q[7],q[6];
cx q[5],q[0];
cx q[4],q[1];
cx q[2],q[3];
cx q[7],q[8];
cx q[2],q[1];
cx q[0],q[5];
rx(pi) q[0];
cx q[0],q[5];
cx q[3],q[2];
cx q[6],q[5];
cx q[2],q[1];
cx q[5],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[2],q[1];
cx q[6],q[5];
cx q[3],q[2];
cx q[5],q[3];
cx q[8],q[3];
rz(3*pi/2) q[3];
cx q[8],q[3];
cx q[5],q[3];
cx q[6],q[4];
cx q[4],q[1];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[6],q[4];
cx q[5],q[8];
cx q[0],q[5];
rx(pi/4) q[0];
cx q[0],q[5];
cx q[5],q[8];
cx q[3],q[4];
cx q[3],q[8];
cx q[2],q[3];
rx(pi/2) q[2];
cx q[2],q[3];
cx q[3],q[8];
cx q[3],q[4];
cx q[7],q[4];
cx q[5],q[4];
cx q[3],q[2];
cx q[4],q[1];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[4];
cx q[6],q[7];
cx q[7],q[2];
rz(7*pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[5],q[6];
cx q[3],q[5];
rx(pi/2) q[3];
cx q[3],q[5];
cx q[5],q[6];
cx q[3],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[0];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[6],q[5];
cx q[7],q[6];
cx q[8],q[7];
cx q[3],q[8];
cx q[4],q[7];
cx q[1],q[4];
cx q[1],q[2];
cx q[0],q[1];
rx(pi/4) q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[1],q[4];
cx q[4],q[7];
cx q[2],q[3];
cx q[5],q[6];
cx q[1],q[2];
cx q[0],q[5];
cx q[0],q[1];
rx(3*pi/2) q[0];
cx q[0],q[1];
cx q[0],q[5];
cx q[1],q[2];
cx q[5],q[6];
cx q[2],q[3];
cx q[3],q[2];
rz(pi/2) q[2];
cx q[3],q[2];
rz(7*pi/4) q[4];
cx q[6],q[5];
cx q[4],q[1];
cx q[5],q[0];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[4],q[1];
cx q[6],q[5];
rx(7*pi/4) q[5];
cx q[0],q[5];
rx(pi) q[0];
cx q[0],q[5];
cx q[3],q[2];
cx q[6],q[5];
cx q[2],q[1];
cx q[5],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[2],q[1];
cx q[6],q[5];
cx q[3],q[2];
cx q[5],q[3];
cx q[8],q[3];
rz(3*pi/2) q[3];
cx q[8],q[3];
cx q[5],q[3];
cx q[6],q[4];
cx q[4],q[1];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[6],q[4];
cx q[5],q[8];
cx q[0],q[5];
rx(pi/4) q[0];
cx q[0],q[5];
cx q[5],q[8];
cx q[3],q[4];
cx q[3],q[8];
cx q[2],q[3];
rx(pi/2) q[2];
cx q[2],q[3];
cx q[3],q[8];
cx q[3],q[4];
cx q[7],q[4];
cx q[5],q[4];
cx q[3],q[2];
cx q[4],q[1];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[4];
cx q[6],q[7];
cx q[7],q[2];
rz(7*pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[5],q[6];
cx q[3],q[5];
rx(pi/2) q[3];
cx q[3],q[5];
cx q[5],q[6];
cx q[3],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[0];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[6],q[5];
cx q[7],q[6];
cx q[8],q[7];
cx q[3],q[8];
cx q[4],q[7];
cx q[1],q[4];
cx q[1],q[2];
cx q[0],q[1];
rx(pi/4) q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[1],q[4];
cx q[4],q[7];
cx q[2],q[3];
cx q[5],q[6];
cx q[1],q[2];
cx q[0],q[5];
cx q[0],q[1];
rx(3*pi/2) q[0];
cx q[0],q[1];
cx q[0],q[5];
cx q[1],q[2];
cx q[5],q[6];
cx q[2],q[3];
cx q[3],q[2];
rz(pi/2) q[2];
cx q[3],q[2];
rz(7*pi/4) q[4];
cx q[6],q[5];
cx q[4],q[1];
cx q[5],q[0];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[4],q[1];
cx q[6],q[5];
rx(7*pi/4) q[5];
cx q[0],q[5];
rx(pi) q[0];
cx q[0],q[5];
cx q[3],q[2];
cx q[6],q[5];
cx q[2],q[1];
cx q[5],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[2],q[1];
cx q[6],q[5];
cx q[3],q[2];
cx q[5],q[3];
cx q[8],q[3];
rz(3*pi/2) q[3];
cx q[8],q[3];
cx q[5],q[3];
cx q[6],q[4];
cx q[4],q[1];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[6],q[4];
cx q[5],q[8];
cx q[0],q[5];
rx(pi/4) q[0];
cx q[0],q[5];
cx q[5],q[8];
cx q[3],q[4];
cx q[3],q[8];
cx q[2],q[3];
rx(pi/2) q[2];
cx q[2],q[3];
cx q[3],q[8];
cx q[3],q[4];
cx q[7],q[4];
cx q[5],q[4];
cx q[3],q[2];
cx q[4],q[1];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[4];
cx q[6],q[7];
cx q[7],q[2];
rz(7*pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[5],q[6];
cx q[3],q[5];
rx(pi/2) q[3];
cx q[3],q[5];
cx q[5],q[6];
cx q[3],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[0];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[6],q[5];
cx q[7],q[6];
cx q[8],q[7];
cx q[3],q[8];
cx q[4],q[7];
cx q[1],q[4];
cx q[1],q[2];
cx q[0],q[1];
rx(pi/4) q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[1],q[4];
cx q[4],q[7];
cx q[2],q[3];
cx q[5],q[6];
cx q[1],q[2];
cx q[0],q[5];
cx q[0],q[1];
rx(3*pi/2) q[0];
cx q[0],q[1];
cx q[0],q[5];
cx q[1],q[2];
cx q[5],q[6];
cx q[2],q[3];
cx q[3],q[2];
rz(pi/2) q[2];
cx q[3],q[2];
rz(7*pi/4) q[4];
cx q[6],q[5];
cx q[4],q[1];
cx q[5],q[0];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[4],q[1];
cx q[6],q[5];
rx(7*pi/4) q[5];
cx q[0],q[5];
rx(pi) q[0];
cx q[0],q[5];
cx q[3],q[2];
cx q[6],q[5];
cx q[2],q[1];
cx q[5],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[2],q[1];
cx q[6],q[5];
cx q[3],q[2];
cx q[5],q[3];
cx q[8],q[3];
rz(3*pi/2) q[3];
cx q[8],q[3];
cx q[5],q[3];
cx q[6],q[4];
cx q[4],q[1];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[6],q[4];
cx q[5],q[8];
cx q[0],q[5];
rx(pi/4) q[0];
cx q[0],q[5];
cx q[5],q[8];
cx q[3],q[4];
cx q[3],q[8];
cx q[2],q[3];
rx(pi/2) q[2];
cx q[2],q[3];
cx q[3],q[8];
cx q[3],q[4];
cx q[7],q[4];
cx q[5],q[4];
cx q[3],q[2];
cx q[4],q[1];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[4];
cx q[6],q[7];
cx q[7],q[2];
rz(7*pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[5],q[6];
cx q[3],q[5];
rx(pi/2) q[3];
cx q[3],q[5];
cx q[5],q[6];
cx q[3],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[0];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[6],q[5];
cx q[7],q[6];
cx q[8],q[7];
cx q[3],q[8];
cx q[4],q[7];
cx q[1],q[4];
cx q[1],q[2];
cx q[0],q[1];
rx(pi/4) q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[1],q[4];
cx q[4],q[7];
cx q[2],q[3];
cx q[5],q[6];
cx q[1],q[2];
cx q[0],q[5];
cx q[0],q[1];
rx(3*pi/2) q[0];
cx q[0],q[1];
cx q[0],q[5];
cx q[1],q[2];
cx q[5],q[6];
cx q[2],q[3];
cx q[3],q[2];
rz(pi/2) q[2];
cx q[3],q[2];
rz(7*pi/4) q[4];
cx q[6],q[5];
cx q[4],q[1];
cx q[5],q[0];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[4],q[1];
cx q[6],q[5];
rx(7*pi/4) q[5];
cx q[0],q[5];
rx(pi) q[0];
cx q[0],q[5];
cx q[3],q[2];
cx q[6],q[5];
cx q[2],q[1];
cx q[5],q[0];
cx q[1],q[0];
rz(pi/2) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[2],q[1];
cx q[6],q[5];
cx q[3],q[2];
cx q[5],q[3];
cx q[8],q[3];
rz(3*pi/2) q[3];
cx q[8],q[3];
cx q[5],q[3];
cx q[6],q[4];
cx q[4],q[1];
cx q[2],q[1];
rz(5*pi/4) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[6],q[4];
cx q[5],q[8];
cx q[0],q[5];
rx(pi/4) q[0];
cx q[0],q[5];
cx q[5],q[8];
cx q[3],q[4];
cx q[3],q[8];
cx q[2],q[3];
rx(pi/2) q[2];
cx q[2],q[3];
cx q[3],q[8];
cx q[3],q[4];
cx q[7],q[4];
cx q[5],q[4];
cx q[3],q[2];
cx q[4],q[1];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
cx q[4],q[1];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[4];
cx q[6],q[7];
cx q[7],q[2];
rz(7*pi/4) q[2];
cx q[7],q[2];
cx q[6],q[7];
cx q[5],q[6];
cx q[3],q[5];
rx(pi/2) q[3];
cx q[3],q[5];
cx q[5],q[6];
cx q[3],q[8];
cx q[8],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[0];
cx q[1],q[0];
rz(7*pi/4) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[6],q[5];
cx q[7],q[6];
cx q[8],q[7];
cx q[3],q[8];
cx q[4],q[7];
cx q[1],q[4];
cx q[1],q[2];
cx q[0],q[1];
rx(pi/4) q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[1],q[4];
cx q[4],q[7];
cx q[2],q[3];
cx q[5],q[6];
cx q[1],q[2];
cx q[0],q[5];
cx q[0],q[1];
rx(3*pi/2) q[0];
cx q[0],q[1];
cx q[0],q[5];
cx q[1],q[2];
cx q[5],q[6];
cx q[2],q[3];
cx q[3],q[2];
rz(pi/2) q[2];
cx q[3],q[2];
rz(7*pi/4) q[4];
cx q[6],q[5];
cx q[4],q[1];
cx q[5],q[0];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[5],q[0];
cx q[4],q[1];
cx q[6],q[5];
rx(7*pi/4) q[5];
cx q[2],q[1];
cx q[2],q[3];
cx q[7],q[8];
cx q[3],q[8];
cx q[7],q[6];
cx q[5],q[0];
cx q[4],q[1];
