OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[7],q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[6];
rz(pi/2) q[8];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[21];
rz(5*pi/4) q[22];
rz(pi/2) q[23];
rz(pi/2) q[24];
cx q[0],q[7];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[3];
rx(pi/2) q[6];
rx(pi/2) q[8];
rx(pi/2) q[14];
rx(pi/2) q[15];
rx(pi/2) q[19];
rx(pi/2) q[20];
rx(pi/2) q[21];
rx(pi/2) q[23];
rx(pi/2) q[24];
cx q[7],q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[3];
rz(pi/2) q[6];
rz(pi/2) q[8];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
cx q[1],q[21];
cx q[2],q[23];
rz(pi/2) q[3];
rz(pi/2) q[6];
rz(3*pi/2) q[8];
rz(pi/2) q[14];
rz(pi/2) q[19];
rz(pi/2) q[20];
rz(pi/2) q[24];
rx(pi/2) q[0];
rx(pi/2) q[3];
rx(pi/2) q[6];
rz(pi/2) q[8];
rx(pi/2) q[14];
rx(pi/2) q[19];
rx(pi/2) q[20];
rz(pi/2) q[21];
rz(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[6];
rx(pi/2) q[8];
rz(pi/2) q[14];
rz(pi/2) q[19];
rz(pi/2) q[20];
rx(pi/2) q[21];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
cx q[19],q[3];
cx q[4],q[14];
cx q[7],q[20];
rz(pi/2) q[8];
rz(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[4];
rz(pi/2) q[14];
rz(pi/2) q[20];
rz(pi/2) q[21];
rz(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[0];
rx(pi/2) q[3];
rx(pi/2) q[4];
rx(pi/2) q[14];
rx(pi/2) q[20];
rx(pi/2) q[21];
rx(pi/2) q[23];
rz(pi/2) q[24];
cx q[7],q[0];
rz(pi/2) q[3];
rz(pi/2) q[4];
rz(pi/2) q[14];
cx q[15],q[24];
rz(pi/2) q[20];
rz(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(3*pi/4) q[4];
rz(pi/2) q[7];
rz(pi/2) q[14];
rz(3*pi/4) q[15];
rz(pi/2) q[20];
rz(3*pi/4) q[21];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[0];
rx(pi/2) q[3];
rz(pi/2) q[4];
rx(pi/2) q[7];
rx(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[20];
rz(pi/2) q[21];
rx(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rx(pi/2) q[4];
rz(pi/2) q[7];
rz(pi/2) q[14];
rx(pi/2) q[15];
rz(pi/2) q[20];
rx(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rz(5*pi/4) q[3];
rz(pi/2) q[4];
cx q[6],q[23];
rz(pi/2) q[15];
rz(pi/2) q[21];
rz(pi/2) q[24];
rx(pi/2) q[0];
rz(pi/2) q[3];
cx q[4],q[14];
rz(pi/2) q[21];
rz(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[0];
rx(pi/2) q[3];
rz(pi/2) q[14];
rx(pi/2) q[21];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rx(pi/2) q[14];
cx q[15],q[24];
rz(pi/2) q[21];
rz(pi/2) q[23];
rx(pi/2) q[0];
cx q[1],q[21];
rz(pi/4) q[3];
rz(pi/2) q[14];
rz(3*pi/4) q[15];
rz(pi/4) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/4) q[1];
rz(pi/2) q[3];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[21];
rz(pi/2) q[23];
rx(pi/2) q[24];
cx q[7],q[0];
rz(pi/2) q[1];
rx(pi/2) q[3];
rx(pi/2) q[14];
rx(pi/2) q[15];
rx(pi/2) q[21];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[3];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(5*pi/4) q[3];
rz(3*pi/4) q[15];
rz(pi/2) q[21];
rz(7*pi/4) q[23];
rx(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[15];
rx(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(5*pi/4) q[0];
rx(pi/2) q[3];
rx(pi/2) q[15];
rz(pi/2) q[21];
rx(pi/2) q[23];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[15];
rz(pi/2) q[21];
rz(pi/2) q[23];
rx(pi/2) q[0];
rz(pi/4) q[3];
cx q[15],q[24];
rx(pi/2) q[21];
rz(pi/4) q[23];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(3*pi/4) q[15];
rz(pi/2) q[21];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/4) q[0];
rx(pi/2) q[3];
rz(pi/2) q[15];
rx(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rx(pi/2) q[15];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[0];
rz(5*pi/4) q[3];
rz(pi/2) q[15];
rz(7*pi/4) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(3*pi/4) q[15];
rz(pi/2) q[23];
rx(pi/2) q[24];
rz(5*pi/4) q[0];
rx(pi/2) q[3];
rz(pi/2) q[15];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rx(pi/2) q[15];
rz(pi/2) q[23];
rx(pi/2) q[0];
rz(pi/4) q[3];
rz(pi/2) q[15];
rz(pi/4) q[23];
rz(pi/2) q[0];
rz(pi/2) q[3];
cx q[15],q[24];
rz(pi/2) q[23];
rz(pi/4) q[0];
rx(pi/2) q[3];
rz(3*pi/4) q[15];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[15];
rz(pi/2) q[23];
rx(pi/2) q[24];
rx(pi/2) q[0];
rz(5*pi/4) q[3];
rx(pi/2) q[15];
rz(7*pi/4) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[15];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(5*pi/4) q[0];
rx(pi/2) q[3];
rz(3*pi/4) q[15];
rx(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[15];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[0];
rz(pi/4) q[3];
rx(pi/2) q[15];
rz(pi/4) q[23];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[15];
rz(pi/2) q[23];
rz(pi/4) q[0];
rx(pi/2) q[3];
cx q[15],q[24];
rx(pi/2) q[23];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(3*pi/4) q[15];
rz(pi/2) q[23];
rz(pi/2) q[24];
rx(pi/2) q[0];
rz(5*pi/4) q[3];
rz(pi/2) q[15];
rz(7*pi/4) q[23];
rx(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rx(pi/2) q[15];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(5*pi/4) q[0];
rx(pi/2) q[3];
rz(pi/2) q[15];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rz(pi/2) q[3];
rz(3*pi/4) q[15];
rz(pi/2) q[23];
rx(pi/2) q[24];
rx(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[15];
rz(pi/4) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rx(pi/2) q[3];
rx(pi/2) q[15];
rz(pi/2) q[23];
rz(pi/4) q[0];
rz(pi/2) q[3];
rz(pi/2) q[15];
rx(pi/2) q[23];
rz(pi/2) q[0];
cx q[19],q[3];
cx q[15],q[24];
rz(pi/2) q[23];
rx(pi/2) q[0];
rz(pi/2) q[3];
rz(3*pi/4) q[15];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rx(pi/2) q[3];
rz(pi/2) q[15];
rx(pi/2) q[23];
rx(pi/2) q[24];
rz(5*pi/4) q[0];
rz(pi/2) q[3];
rx(pi/2) q[15];
rz(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
cx q[2],q[23];
rz(pi/4) q[3];
rz(pi/2) q[15];
rz(pi/2) q[24];
rx(pi/2) q[0];
rz(pi/2) q[3];
rz(pi/2) q[23];
rx(pi/2) q[24];
rz(pi/2) q[0];
rx(pi/2) q[3];
rx(pi/2) q[23];
rz(pi/2) q[24];
cx q[0],q[7];
rz(pi/2) q[3];
cx q[15],q[24];
rz(pi/2) q[23];
rz(pi/4) q[0];
rz(pi/2) q[7];
rz(pi/2) q[15];
rz(7*pi/4) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
rx(pi/2) q[7];
rx(pi/2) q[15];
rz(pi/2) q[23];
rx(pi/2) q[24];
rx(pi/2) q[0];
rz(pi/2) q[7];
rz(pi/2) q[15];
rx(pi/2) q[23];
rz(pi/2) q[24];
rz(pi/2) q[0];
cx q[15],q[2];
rz(pi/2) q[23];
rz(3*pi/2) q[24];
cx q[0],q[19];
cx q[2],q[15];
rz(pi/2) q[23];
cx q[19],q[0];
cx q[15],q[2];
rx(pi/2) q[23];
cx q[0],q[19];
rz(pi/2) q[15];
rz(pi/2) q[23];
cx q[6],q[23];
rx(pi/2) q[15];
cx q[19],q[20];
rz(pi/2) q[15];
rz(pi/2) q[20];
rz(pi/2) q[23];
cx q[6],q[15];
rx(pi/2) q[20];
rx(pi/2) q[23];
rz(pi/2) q[15];
rz(pi/2) q[20];
rz(pi/2) q[23];
rx(pi/2) q[15];
rz(pi/2) q[20];
rz(pi/2) q[23];
rz(pi/2) q[15];
rx(pi/2) q[20];
rx(pi/2) q[23];
rz(3*pi/2) q[15];
rz(pi/2) q[20];
rz(pi/2) q[23];
rz(pi/2) q[15];
rx(pi/2) q[15];
rz(pi/2) q[15];
