OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
rz(pi/2) q[0];
cx q[2],q[22];
rz(pi/2) q[5];
rz(pi/2) q[6];
rz(pi/2) q[9];
rz(pi/2) q[13];
cx q[14],q[18];
rz(pi/2) q[15];
rz(pi/2) q[16];
rz(pi/2) q[17];
rz(pi/2) q[19];
rz(pi/2) q[21];
rx(pi/2) q[0];
cx q[22],q[2];
rx(pi/2) q[5];
rx(pi/2) q[6];
rx(pi/2) q[9];
rx(pi/2) q[13];
cx q[18],q[14];
rx(pi/2) q[15];
rx(pi/2) q[16];
rx(pi/2) q[17];
rx(pi/2) q[19];
rx(pi/2) q[21];
rz(pi/2) q[0];
cx q[2],q[22];
rz(pi/2) q[5];
rz(pi/2) q[6];
rz(pi/2) q[9];
rz(pi/2) q[13];
cx q[14],q[18];
rz(pi/2) q[15];
rz(pi/2) q[16];
rz(pi/2) q[17];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[2];
rz(pi/2) q[5];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(3*pi/2) q[15];
rz(pi/2) q[16];
rz(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[21];
rz(pi/2) q[22];
rx(pi/2) q[2];
rx(pi/2) q[5];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[15];
rx(pi/2) q[16];
rx(pi/2) q[17];
rx(pi/2) q[18];
rx(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[2];
rz(pi/2) q[5];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[15];
rz(pi/2) q[16];
rz(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi) q[2];
cx q[5],q[13];
cx q[11],q[16];
rz(pi/2) q[15];
cx q[18],q[21];
cx q[19],q[22];
rz(pi/2) q[2];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[16];
rz(pi/4) q[18];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[22];
rx(pi/2) q[2];
rz(pi/2) q[11];
rx(pi/2) q[13];
rx(pi/2) q[16];
rz(pi/2) q[18];
rx(pi/2) q[19];
rx(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[2];
rx(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[16];
rx(pi/2) q[18];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[11];
rz(pi/2) q[13];
rz(pi/2) q[16];
rz(pi/2) q[18];
rz(pi/2) q[21];
rz(pi/2) q[22];
rx(pi/2) q[13];
rx(pi/2) q[16];
rz(pi/2) q[18];
rx(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[13];
rz(pi/2) q[16];
rx(pi/2) q[18];
rz(pi/2) q[21];
rz(pi/2) q[22];
cx q[11],q[16];
cx q[13],q[17];
rz(pi/2) q[18];
rz(pi/2) q[22];
rz(pi/2) q[11];
rz(3*pi/4) q[13];
cx q[14],q[18];
rz(pi/2) q[16];
rz(pi/2) q[17];
rx(pi/2) q[22];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[16];
rx(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[22];
cx q[2],q[22];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[17];
rx(pi/2) q[18];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[16];
rz(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[22];
rz(5*pi/4) q[13];
rx(pi/2) q[16];
rx(pi/2) q[17];
rz(7*pi/4) q[18];
rx(pi/2) q[22];
rz(pi/2) q[13];
rz(pi/2) q[16];
rz(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[22];
rx(pi/2) q[13];
rx(pi/2) q[18];
rz(3*pi/4) q[22];
rz(pi/2) q[13];
rz(pi/2) q[18];
rz(pi/2) q[22];
rz(5*pi/4) q[13];
rz(pi/2) q[18];
rx(pi/2) q[22];
rz(pi/2) q[13];
rx(pi/2) q[18];
rz(pi/2) q[22];
rx(pi/2) q[13];
rz(pi/2) q[18];
rz(pi/4) q[22];
rz(pi/2) q[13];
cx q[14],q[18];
rz(pi/2) q[22];
rz(5*pi/4) q[13];
rz(3*pi/4) q[14];
rz(pi/2) q[18];
rx(pi/2) q[22];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[18];
rz(pi/2) q[22];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[18];
rz(7*pi/4) q[22];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[22];
rz(pi/2) q[13];
cx q[14],q[21];
rx(pi/2) q[22];
rx(pi/2) q[13];
rz(pi/4) q[14];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[21];
rz(pi/4) q[22];
cx q[5],q[13];
rx(pi/2) q[14];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[5];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[21];
rx(pi/2) q[22];
rx(pi/2) q[5];
rx(pi/2) q[13];
rz(3*pi/4) q[14];
rx(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[5];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[21];
rz(7*pi/4) q[22];
cx q[0],q[5];
rz(5*pi/4) q[13];
rx(pi/2) q[14];
rz(pi/2) q[22];
rz(pi/2) q[0];
rz(pi/2) q[5];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[22];
rz(pi/2) q[0];
rx(pi/2) q[5];
rx(pi/2) q[13];
cx q[14],q[18];
rz(pi/2) q[22];
rx(pi/2) q[0];
rz(pi/2) q[5];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/4) q[22];
rz(pi/2) q[0];
cx q[13],q[17];
rx(pi/2) q[14];
rz(pi/2) q[22];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[17];
rx(pi/2) q[22];
rx(pi/2) q[13];
rz(7*pi/4) q[14];
rx(pi/2) q[17];
rz(pi/2) q[22];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(pi/2) q[17];
rz(7*pi/4) q[22];
rx(pi/2) q[14];
rz(pi/2) q[17];
rz(pi/2) q[22];
rz(pi/2) q[14];
rx(pi/2) q[17];
rx(pi/2) q[22];
cx q[14],q[18];
rz(pi/2) q[17];
rz(pi/2) q[22];
cx q[5],q[17];
cx q[14],q[21];
rz(pi/4) q[22];
cx q[5],q[13];
rz(pi/4) q[14];
rz(pi/2) q[17];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[5];
rz(pi/2) q[13];
rz(pi/2) q[14];
rx(pi/2) q[17];
rx(pi/2) q[21];
rx(pi/2) q[22];
rx(pi/2) q[5];
rx(pi/2) q[13];
rx(pi/2) q[14];
rz(pi/2) q[17];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[5];
rz(pi/2) q[13];
rz(pi/2) q[14];
rz(3*pi/4) q[17];
rz(pi/2) q[21];
rz(7*pi/4) q[22];
cx q[0],q[5];
rz(pi/2) q[13];
rz(3*pi/4) q[14];
rz(pi/2) q[17];
rx(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[0];
rz(pi/2) q[5];
rz(pi/2) q[14];
rx(pi/2) q[17];
rz(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[0];
rx(pi/2) q[5];
rx(pi/2) q[14];
rz(pi/2) q[17];
rz(pi/2) q[22];
rx(pi/2) q[0];
cx q[22],q[2];
rz(pi/2) q[5];
rz(pi/2) q[14];
rz(pi/2) q[0];
rz(pi/2) q[5];
cx q[14],q[18];
rz(pi/4) q[22];
rz(pi/2) q[14];
rz(pi/2) q[22];
rx(pi/2) q[14];
rx(pi/2) q[22];
rz(pi/2) q[14];
rz(pi/2) q[22];
rz(7*pi/4) q[14];
rz(pi/2) q[22];
rz(pi/2) q[14];
rx(pi/2) q[22];
rx(pi/2) q[14];
rz(pi/2) q[22];
cx q[2],q[22];
rz(pi/2) q[14];
cx q[2],q[19];
cx q[14],q[21];
rz(pi/2) q[22];
rz(pi/4) q[14];
rz(pi/2) q[19];
rz(pi/2) q[21];
rx(pi/2) q[22];
rz(pi/2) q[14];
rx(pi/2) q[19];
rx(pi/2) q[21];
rz(pi/2) q[22];
rx(pi/2) q[14];
rz(pi/2) q[19];
rz(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[14];
rz(pi) q[19];
rz(pi/2) q[21];
rx(pi/2) q[22];
rz(7*pi/4) q[14];
rz(pi/2) q[19];
rx(pi/2) q[21];
rz(pi/2) q[22];
rz(pi/2) q[14];
rx(pi/2) q[19];
rz(pi/2) q[21];
rx(pi/2) q[14];
rz(pi/2) q[19];
rz(pi/2) q[14];
cx q[14],q[18];
cx q[9],q[18];
rz(pi/2) q[14];
cx q[6],q[18];
rz(pi/2) q[9];
rx(pi/2) q[14];
rx(pi/2) q[9];
rz(pi/2) q[14];
rz(pi/2) q[9];
rz(3*pi/4) q[14];
rz(pi/2) q[9];
rz(pi/2) q[14];
rx(pi/2) q[9];
rx(pi/2) q[14];
rz(pi/2) q[9];
rz(pi/2) q[14];
cx q[6],q[9];
cx q[14],q[21];
rz(3*pi/2) q[6];
rz(pi/2) q[9];
rz(pi/4) q[14];
rz(pi/2) q[21];
rz(pi/2) q[6];
rx(pi/2) q[9];
rz(pi/2) q[14];
rx(pi/2) q[21];
rx(pi/2) q[6];
rz(pi/2) q[9];
rx(pi/2) q[14];
rz(pi/2) q[21];
rz(pi/2) q[6];
rz(pi/2) q[9];
rz(pi/2) q[14];
rz(3*pi/2) q[21];
rx(pi/2) q[9];
rz(3*pi/4) q[14];
rz(pi/2) q[21];
rz(pi/2) q[9];
rz(pi/2) q[14];
rx(pi/2) q[21];
cx q[6],q[9];
rx(pi/2) q[14];
rz(pi/2) q[21];
rz(pi/2) q[9];
rz(pi/2) q[14];
rx(pi/2) q[9];
cx q[14],q[18];
rz(pi/2) q[9];
rz(pi/2) q[14];
rz(pi/2) q[18];
rz(7*pi/4) q[9];
rx(pi/2) q[14];
rx(pi/2) q[18];
rz(pi/2) q[9];
rz(pi/2) q[14];
rz(pi/2) q[18];
rx(pi/2) q[9];
rz(7*pi/4) q[14];
rz(pi/2) q[18];
rz(pi/2) q[9];
rx(pi/2) q[18];
rz(pi/2) q[18];
cx q[9],q[18];
rz(pi/2) q[9];
rz(pi/2) q[18];
rx(pi/2) q[9];
rx(pi/2) q[18];
rz(pi/2) q[9];
rz(pi/2) q[18];
rz(pi/2) q[18];
rx(pi/2) q[18];
rz(pi/2) q[18];
cx q[6],q[18];
rz(3*pi/2) q[6];
rz(pi/2) q[18];
rz(pi/2) q[6];
rx(pi/2) q[18];
rx(pi/2) q[6];
rz(pi/2) q[18];
rz(pi/2) q[6];