OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[4],q[8];
cx q[4],q[15];
cx q[4],q[9];
rx(pi/4) q[4];
cx q[4],q[9];
cx q[4],q[15];
cx q[4],q[8];
cx q[1],q[9];
cx q[1],q[13];
rx(7*pi/4) q[1];
cx q[1],q[13];
cx q[1],q[9];
rz(pi) q[6];
cx q[12],q[4];
rz(pi/4) q[4];
cx q[12],q[4];
cx q[6],q[3];
cx q[10],q[3];
cx q[12],q[3];
rz(5*pi/4) q[3];
cx q[12],q[3];
cx q[10],q[3];
cx q[6],q[3];
cx q[2],q[13];
cx q[2],q[5];
cx q[2],q[9];
rx(3*pi/2) q[2];
cx q[2],q[9];
cx q[2],q[5];
cx q[2],q[13];
cx q[6],q[3];
cx q[5],q[3];
cx q[12],q[3];
rz(7*pi/4) q[3];
cx q[12],q[3];
cx q[5],q[3];
cx q[6],q[3];
cx q[0],q[6];
cx q[0],q[3];
rx(5*pi/4) q[0];
cx q[0],q[3];
cx q[0],q[6];
rz(5*pi/4) q[6];
cx q[0],q[14];
rx(3*pi/2) q[0];
cx q[0],q[14];
rx(5*pi/4) q[14];
rx(3*pi/4) q[15];
cx q[11],q[3];
cx q[5],q[3];
cx q[12],q[3];
rz(pi/2) q[3];
cx q[12],q[3];
cx q[5],q[3];
cx q[11],q[3];
cx q[7],q[13];
cx q[7],q[10];
rx(7*pi/4) q[7];
cx q[7],q[10];
cx q[7],q[13];
cx q[4],q[0];
rz(5*pi/4) q[0];
cx q[4],q[0];
rx(5*pi/4) q[7];
cx q[4],q[8];
cx q[4],q[15];
cx q[4],q[9];
rx(pi/4) q[4];
cx q[4],q[9];
cx q[4],q[15];
cx q[4],q[8];
cx q[1],q[9];
cx q[1],q[13];
rx(7*pi/4) q[1];
cx q[1],q[13];
cx q[1],q[9];
rz(pi) q[6];
cx q[12],q[4];
rz(pi/4) q[4];
cx q[12],q[4];
cx q[6],q[3];
cx q[10],q[3];
cx q[12],q[3];
rz(5*pi/4) q[3];
cx q[12],q[3];
cx q[10],q[3];
cx q[6],q[3];
cx q[2],q[13];
cx q[2],q[5];
cx q[2],q[9];
rx(3*pi/2) q[2];
cx q[2],q[9];
cx q[2],q[5];
cx q[2],q[13];
cx q[6],q[3];
cx q[5],q[3];
cx q[12],q[3];
rz(7*pi/4) q[3];
cx q[12],q[3];
cx q[5],q[3];
cx q[6],q[3];
cx q[0],q[6];
cx q[0],q[3];
rx(5*pi/4) q[0];
cx q[0],q[3];
cx q[0],q[6];
rz(5*pi/4) q[6];
cx q[0],q[14];
rx(3*pi/2) q[0];
cx q[0],q[14];
rx(5*pi/4) q[14];
rx(3*pi/4) q[15];
cx q[11],q[3];
cx q[5],q[3];
cx q[12],q[3];
rz(pi/2) q[3];
cx q[12],q[3];
cx q[5],q[3];
cx q[11],q[3];
cx q[7],q[13];
cx q[7],q[10];
rx(7*pi/4) q[7];
cx q[7],q[10];
cx q[7],q[13];
cx q[4],q[0];
rz(5*pi/4) q[0];
cx q[4],q[0];
rx(5*pi/4) q[7];
cx q[4],q[8];
cx q[4],q[15];
cx q[4],q[9];
rx(pi/4) q[4];
cx q[4],q[9];
cx q[4],q[15];
cx q[4],q[8];
cx q[1],q[9];
cx q[1],q[13];
rx(7*pi/4) q[1];
cx q[1],q[13];
cx q[1],q[9];
rz(pi) q[6];
cx q[12],q[4];
rz(pi/4) q[4];
cx q[12],q[4];
cx q[6],q[3];
cx q[10],q[3];
cx q[12],q[3];
rz(5*pi/4) q[3];
cx q[12],q[3];
cx q[10],q[3];
cx q[6],q[3];
cx q[2],q[13];
cx q[2],q[5];
cx q[2],q[9];
rx(3*pi/2) q[2];
cx q[2],q[9];
cx q[2],q[5];
cx q[2],q[13];
cx q[6],q[3];
cx q[5],q[3];
cx q[12],q[3];
rz(7*pi/4) q[3];
cx q[12],q[3];
cx q[5],q[3];
cx q[6],q[3];
cx q[0],q[6];
cx q[0],q[3];
rx(5*pi/4) q[0];
cx q[0],q[3];
cx q[0],q[6];
rz(5*pi/4) q[6];
cx q[0],q[14];
rx(3*pi/2) q[0];
cx q[0],q[14];
rx(5*pi/4) q[14];
rx(3*pi/4) q[15];
cx q[11],q[3];
cx q[5],q[3];
cx q[12],q[3];
rz(pi/2) q[3];
cx q[12],q[3];
cx q[5],q[3];
cx q[11],q[3];
cx q[7],q[13];
cx q[7],q[10];
rx(7*pi/4) q[7];
cx q[7],q[10];
cx q[7],q[13];
cx q[4],q[0];
rz(5*pi/4) q[0];
cx q[4],q[0];
rx(5*pi/4) q[7];
cx q[4],q[8];
cx q[4],q[15];
cx q[4],q[9];
rx(pi/4) q[4];
cx q[4],q[9];
cx q[4],q[15];
cx q[4],q[8];
cx q[1],q[9];
cx q[1],q[13];
rx(7*pi/4) q[1];
cx q[1],q[13];
cx q[1],q[9];
rz(pi) q[6];
cx q[12],q[4];
rz(pi/4) q[4];
cx q[12],q[4];
cx q[6],q[3];
cx q[10],q[3];
cx q[12],q[3];
rz(5*pi/4) q[3];
cx q[12],q[3];
cx q[10],q[3];
cx q[6],q[3];
cx q[2],q[13];
cx q[2],q[5];
cx q[2],q[9];
rx(3*pi/2) q[2];
cx q[2],q[9];
cx q[2],q[5];
cx q[2],q[13];
cx q[6],q[3];
cx q[5],q[3];
cx q[12],q[3];
rz(7*pi/4) q[3];
cx q[12],q[3];
cx q[5],q[3];
cx q[6],q[3];
cx q[0],q[6];
cx q[0],q[3];
rx(5*pi/4) q[0];
cx q[0],q[3];
cx q[0],q[6];
rz(5*pi/4) q[6];
cx q[0],q[14];
rx(3*pi/2) q[0];
cx q[0],q[14];
rx(5*pi/4) q[14];
rx(3*pi/4) q[15];
cx q[11],q[3];
cx q[5],q[3];
cx q[12],q[3];
rz(pi/2) q[3];
cx q[12],q[3];
cx q[5],q[3];
cx q[11],q[3];
cx q[7],q[13];
cx q[7],q[10];
rx(7*pi/4) q[7];
cx q[7],q[10];
cx q[7],q[13];
cx q[4],q[0];
rz(5*pi/4) q[0];
cx q[4],q[0];
rx(5*pi/4) q[7];
cx q[4],q[8];
cx q[4],q[15];
cx q[4],q[9];
rx(pi/4) q[4];
cx q[4],q[9];
cx q[4],q[15];
cx q[4],q[8];
cx q[1],q[9];
cx q[1],q[13];
rx(7*pi/4) q[1];
cx q[1],q[13];
cx q[1],q[9];
rz(pi) q[6];
cx q[12],q[4];
rz(pi/4) q[4];
cx q[12],q[4];
cx q[6],q[3];
cx q[10],q[3];
cx q[12],q[3];
rz(5*pi/4) q[3];
cx q[12],q[3];
cx q[10],q[3];
cx q[6],q[3];
cx q[2],q[13];
cx q[2],q[5];
cx q[2],q[9];
rx(3*pi/2) q[2];
cx q[2],q[9];
cx q[2],q[5];
cx q[2],q[13];
cx q[6],q[3];
cx q[5],q[3];
cx q[12],q[3];
rz(7*pi/4) q[3];
cx q[12],q[3];
cx q[5],q[3];
cx q[6],q[3];
cx q[0],q[6];
cx q[0],q[3];
rx(5*pi/4) q[0];
cx q[0],q[3];
cx q[0],q[6];
rz(5*pi/4) q[6];
cx q[0],q[14];
rx(3*pi/2) q[0];
cx q[0],q[14];
rx(5*pi/4) q[14];
rx(3*pi/4) q[15];
cx q[11],q[3];
cx q[5],q[3];
cx q[12],q[3];
rz(pi/2) q[3];
cx q[12],q[3];
cx q[5],q[3];
cx q[11],q[3];
cx q[7],q[13];
cx q[7],q[10];
rx(7*pi/4) q[7];
cx q[7],q[10];
cx q[7],q[13];
cx q[4],q[0];
rz(5*pi/4) q[0];
cx q[4],q[0];
rx(5*pi/4) q[7];