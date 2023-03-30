OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(5*pi/4) q[9];
rz(3*pi/2) q[0];
cx q[14],q[13];
cx q[10],q[13];
cx q[13],q[10];
cx q[10],q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[5],q[2];
cx q[2],q[5];
cx q[10],q[5];
cx q[13],q[10];
cx q[10],q[13];
cx q[14],q[13];
cx q[10],q[5];
cx q[5],q[10];
cx q[5],q[6];
cx q[6],q[5];
cx q[7],q[6];
rx(pi) q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[5],q[10];
cx q[10],q[5];
rx(7*pi/4) q[2];
cx q[8],q[15];
cx q[15],q[8];
cx q[8],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[4];
rz(5*pi/4) q[4];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[9],q[6];
cx q[6],q[9];
cx q[8],q[9];
cx q[15],q[8];
cx q[8],q[15];
rx(pi/2) q[3];
cx q[8],q[9];
cx q[9],q[8];
cx q[10],q[9];
cx q[10],q[5];
cx q[5],q[10];
cx q[4],q[5];
rx(3*pi/4) q[4];
cx q[4],q[5];
cx q[5],q[10];
cx q[10],q[5];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[9];
rz(5*pi/4) q[1];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
rz(5*pi/4) q[9];
rz(3*pi/2) q[0];
cx q[14],q[13];
cx q[10],q[13];
cx q[13],q[10];
cx q[10],q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[5],q[2];
cx q[2],q[5];
cx q[10],q[5];
cx q[13],q[10];
cx q[10],q[13];
cx q[14],q[13];
cx q[10],q[5];
cx q[5],q[10];
cx q[5],q[6];
cx q[6],q[5];
cx q[7],q[6];
rx(pi) q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[5],q[10];
cx q[10],q[5];
rx(7*pi/4) q[2];
cx q[8],q[15];
cx q[15],q[8];
cx q[8],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[4];
rz(5*pi/4) q[4];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[9],q[6];
cx q[6],q[9];
cx q[8],q[9];
cx q[15],q[8];
cx q[8],q[15];
rx(pi/2) q[3];
cx q[8],q[9];
cx q[9],q[8];
cx q[10],q[9];
cx q[10],q[5];
cx q[5],q[10];
cx q[4],q[5];
rx(3*pi/4) q[4];
cx q[4],q[5];
cx q[5],q[10];
cx q[10],q[5];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[9];
rz(5*pi/4) q[1];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
rz(5*pi/4) q[9];
rz(3*pi/2) q[0];
cx q[14],q[13];
cx q[10],q[13];
cx q[13],q[10];
cx q[10],q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[5],q[2];
cx q[2],q[5];
cx q[10],q[5];
cx q[13],q[10];
cx q[10],q[13];
cx q[14],q[13];
cx q[10],q[5];
cx q[5],q[10];
cx q[5],q[6];
cx q[6],q[5];
cx q[7],q[6];
rx(pi) q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[5],q[10];
cx q[10],q[5];
rx(7*pi/4) q[2];
cx q[8],q[15];
cx q[15],q[8];
cx q[8],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[4];
rz(5*pi/4) q[4];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[9],q[6];
cx q[6],q[9];
cx q[8],q[9];
cx q[15],q[8];
cx q[8],q[15];
rx(pi/2) q[3];
cx q[8],q[9];
cx q[9],q[8];
cx q[10],q[9];
cx q[10],q[5];
cx q[5],q[10];
cx q[4],q[5];
rx(3*pi/4) q[4];
cx q[4],q[5];
cx q[5],q[10];
cx q[10],q[5];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[9];
rz(5*pi/4) q[1];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
rz(5*pi/4) q[9];
rz(3*pi/2) q[0];
cx q[14],q[13];
cx q[10],q[13];
cx q[13],q[10];
cx q[10],q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[5],q[2];
cx q[2],q[5];
cx q[10],q[5];
cx q[13],q[10];
cx q[10],q[13];
cx q[14],q[13];
cx q[10],q[5];
cx q[5],q[10];
cx q[5],q[6];
cx q[6],q[5];
cx q[7],q[6];
rx(pi) q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[5],q[10];
cx q[10],q[5];
rx(7*pi/4) q[2];
cx q[8],q[15];
cx q[15],q[8];
cx q[8],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[4];
rz(5*pi/4) q[4];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[9],q[6];
cx q[6],q[9];
cx q[8],q[9];
cx q[15],q[8];
cx q[8],q[15];
rx(pi/2) q[3];
cx q[8],q[9];
cx q[9],q[8];
cx q[10],q[9];
cx q[10],q[5];
cx q[5],q[10];
cx q[4],q[5];
rx(3*pi/4) q[4];
cx q[4],q[5];
cx q[5],q[10];
cx q[10],q[5];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[9];
rz(5*pi/4) q[1];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];
rz(5*pi/4) q[9];
rz(3*pi/2) q[0];
cx q[14],q[13];
cx q[10],q[13];
cx q[13],q[10];
cx q[10],q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(5*pi/4) q[0];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[5],q[2];
cx q[2],q[5];
cx q[10],q[5];
cx q[13],q[10];
cx q[10],q[13];
cx q[14],q[13];
cx q[10],q[5];
cx q[5],q[10];
cx q[5],q[6];
cx q[6],q[5];
cx q[7],q[6];
rx(pi) q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[5],q[10];
cx q[10],q[5];
rx(7*pi/4) q[2];
cx q[8],q[15];
cx q[15],q[8];
cx q[8],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[4];
rz(5*pi/4) q[4];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[9],q[6];
cx q[6],q[9];
cx q[8],q[9];
cx q[15],q[8];
cx q[8],q[15];
rx(pi/2) q[3];
cx q[8],q[9];
cx q[9],q[8];
cx q[10],q[9];
cx q[10],q[5];
cx q[5],q[10];
cx q[4],q[5];
rx(3*pi/4) q[4];
cx q[4],q[5];
cx q[5],q[10];
cx q[10],q[5];
cx q[10],q[9];
cx q[9],q[8];
cx q[8],q[9];
rz(5*pi/4) q[1];
cx q[2],q[1];
rz(3*pi/2) q[1];
cx q[2],q[1];