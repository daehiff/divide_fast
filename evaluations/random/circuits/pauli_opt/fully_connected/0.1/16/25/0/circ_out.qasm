OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[0],q[13];
cx q[3],q[5];
cx q[2],q[19];
cx q[18],q[9];
cx q[5],q[0];
cx q[13],q[17];
cx q[16],q[23];
cx q[0],q[13];
cx q[9],q[6];
cx q[22],q[2];
cx q[11],q[16];
cx q[2],q[19];
rx(pi) q[2];
cx q[2],q[19];
rx(pi) q[21];
cx q[2],q[19];
cx q[2],q[22];
rx(pi) q[2];
cx q[2],q[22];
cx q[2],q[19];
rz(3*pi/2) q[0];
rz(pi/2) q[6];
rz(3*pi/4) q[9];
rz(pi/4) q[13];
rx(7*pi/4) q[14];
cx q[14],q[21];
rx(pi/2) q[14];
cx q[14],q[21];
rx(7*pi/4) q[2];
rx(3*pi/2) q[11];
rx(7*pi/4) q[17];
rx(3*pi/2) q[15];
rx(3*pi/2) q[13];
cx q[18],q[14];
rz(3*pi/4) q[14];
cx q[18],q[14];
rz(7*pi/4) q[14];
rz(pi/4) q[2];
cx q[2],q[19];
rx(pi) q[2];
cx q[2],q[19];
rx(pi) q[21];
cx q[2],q[19];
cx q[2],q[22];
rx(pi) q[2];
cx q[2],q[22];
cx q[2],q[19];
rz(3*pi/2) q[0];
rz(pi/2) q[6];
rz(3*pi/4) q[9];
rz(pi/4) q[13];
rx(7*pi/4) q[14];
cx q[14],q[21];
rx(pi/2) q[14];
cx q[14],q[21];
rx(7*pi/4) q[2];
rx(3*pi/2) q[11];
rx(7*pi/4) q[17];
rx(3*pi/2) q[15];
rx(3*pi/2) q[13];
cx q[18],q[14];
rz(3*pi/4) q[14];
cx q[18],q[14];
rz(7*pi/4) q[14];
rz(pi/4) q[2];
cx q[2],q[19];
rx(pi) q[2];
cx q[2],q[19];
rx(pi) q[21];
cx q[2],q[19];
cx q[2],q[22];
rx(pi) q[2];
cx q[2],q[22];
cx q[2],q[19];
rz(3*pi/2) q[0];
rz(pi/2) q[6];
rz(3*pi/4) q[9];
rz(pi/4) q[13];
rx(7*pi/4) q[14];
cx q[14],q[21];
rx(pi/2) q[14];
cx q[14],q[21];
rx(7*pi/4) q[2];
rx(3*pi/2) q[11];
rx(7*pi/4) q[17];
rx(3*pi/2) q[15];
rx(3*pi/2) q[13];
cx q[18],q[14];
rz(3*pi/4) q[14];
cx q[18],q[14];
rz(7*pi/4) q[14];
rz(pi/4) q[2];
cx q[2],q[19];
rx(pi) q[2];
cx q[2],q[19];
rx(pi) q[21];
cx q[2],q[19];
cx q[2],q[22];
rx(pi) q[2];
cx q[2],q[22];
cx q[2],q[19];
rz(3*pi/2) q[0];
rz(pi/2) q[6];
rz(3*pi/4) q[9];
rz(pi/4) q[13];
rx(7*pi/4) q[14];
cx q[14],q[21];
rx(pi/2) q[14];
cx q[14],q[21];
rx(7*pi/4) q[2];
rx(3*pi/2) q[11];
rx(7*pi/4) q[17];
rx(3*pi/2) q[15];
rx(3*pi/2) q[13];
cx q[18],q[14];
rz(3*pi/4) q[14];
cx q[18],q[14];
rz(7*pi/4) q[14];
rz(pi/4) q[2];
cx q[2],q[19];
rx(pi) q[2];
cx q[2],q[19];
rx(pi) q[21];
cx q[2],q[19];
cx q[2],q[22];
rx(pi) q[2];
cx q[2],q[22];
cx q[2],q[19];
rz(3*pi/2) q[0];
rz(pi/2) q[6];
rz(3*pi/4) q[9];
rz(pi/4) q[13];
rx(7*pi/4) q[14];
cx q[14],q[21];
rx(pi/2) q[14];
cx q[14],q[21];
rx(7*pi/4) q[2];
rx(3*pi/2) q[11];
rx(7*pi/4) q[17];
rx(3*pi/2) q[15];
rx(3*pi/2) q[13];
cx q[18],q[14];
rz(3*pi/4) q[14];
cx q[18],q[14];
rz(7*pi/4) q[14];
rz(pi/4) q[2];
cx q[0],q[13];
cx q[9],q[6];
cx q[22],q[2];
cx q[11],q[16];
cx q[18],q[9];
cx q[5],q[0];
cx q[13],q[17];
cx q[16],q[23];
cx q[0],q[13];
cx q[3],q[5];
cx q[2],q[19];
