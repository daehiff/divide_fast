OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[23],q[16];
cx q[16],q[23];
cx q[15],q[16];
cx q[16],q[23];
cx q[23],q[16];
cx q[6],q[13];
cx q[13],q[6];
rx(pi) q[0];
rx(pi) q[1];
rx(pi) q[4];
rx(pi) q[5];
rx(pi) q[11];
rx(pi) q[24];
cx q[16],q[23];
cx q[23],q[16];
cx q[13],q[16];
cx q[16],q[13];
cx q[6],q[13];
cx q[13],q[6];
cx q[3],q[6];
cx q[6],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[6],q[3];
cx q[3],q[6];
cx q[13],q[6];
cx q[6],q[13];
cx q[16],q[13];
cx q[13],q[16];
cx q[23],q[16];
cx q[16],q[23];
rz(3*pi/4) q[21];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[13];
cx q[13],q[6];
cx q[6],q[3];
rz(pi/2) q[3];
cx q[6],q[3];
cx q[13],q[6];
cx q[12],q[13];
cx q[17],q[12];
cx q[12],q[17];
cx q[8],q[7];
rz(3*pi/4) q[7];
cx q[8],q[7];
rx(pi/4) q[15];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[11],q[12];
cx q[11],q[8];
cx q[8],q[11];
cx q[8],q[1];
cx q[1],q[8];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[8];
cx q[8],q[1];
cx q[8],q[11];
cx q[11],q[8];
cx q[11],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[13],q[14];
rz(5*pi/4) q[11];
rx(pi) q[0];
rx(pi) q[1];
rx(pi) q[4];
rx(pi) q[5];
rx(pi) q[11];
rx(pi) q[24];
cx q[16],q[23];
cx q[23],q[16];
cx q[13],q[16];
cx q[16],q[13];
cx q[6],q[13];
cx q[13],q[6];
cx q[3],q[6];
cx q[6],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[6],q[3];
cx q[3],q[6];
cx q[13],q[6];
cx q[6],q[13];
cx q[16],q[13];
cx q[13],q[16];
cx q[23],q[16];
cx q[16],q[23];
rz(3*pi/4) q[21];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[13];
cx q[13],q[6];
cx q[6],q[3];
rz(pi/2) q[3];
cx q[6],q[3];
cx q[13],q[6];
cx q[12],q[13];
cx q[17],q[12];
cx q[12],q[17];
cx q[8],q[7];
rz(3*pi/4) q[7];
cx q[8],q[7];
rx(pi/4) q[15];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[11],q[12];
cx q[11],q[8];
cx q[8],q[11];
cx q[8],q[1];
cx q[1],q[8];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[8];
cx q[8],q[1];
cx q[8],q[11];
cx q[11],q[8];
cx q[11],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[13],q[14];
rz(5*pi/4) q[11];
rx(pi) q[0];
rx(pi) q[1];
rx(pi) q[4];
rx(pi) q[5];
rx(pi) q[11];
rx(pi) q[24];
cx q[16],q[23];
cx q[23],q[16];
cx q[13],q[16];
cx q[16],q[13];
cx q[6],q[13];
cx q[13],q[6];
cx q[3],q[6];
cx q[6],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[6],q[3];
cx q[3],q[6];
cx q[13],q[6];
cx q[6],q[13];
cx q[16],q[13];
cx q[13],q[16];
cx q[23],q[16];
cx q[16],q[23];
rz(3*pi/4) q[21];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[13];
cx q[13],q[6];
cx q[6],q[3];
rz(pi/2) q[3];
cx q[6],q[3];
cx q[13],q[6];
cx q[12],q[13];
cx q[17],q[12];
cx q[12],q[17];
cx q[8],q[7];
rz(3*pi/4) q[7];
cx q[8],q[7];
rx(pi/4) q[15];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[11],q[12];
cx q[11],q[8];
cx q[8],q[11];
cx q[8],q[1];
cx q[1],q[8];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[8];
cx q[8],q[1];
cx q[8],q[11];
cx q[11],q[8];
cx q[11],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[13],q[14];
rz(5*pi/4) q[11];
rx(pi) q[0];
rx(pi) q[1];
rx(pi) q[4];
rx(pi) q[5];
rx(pi) q[11];
rx(pi) q[24];
cx q[16],q[23];
cx q[23],q[16];
cx q[13],q[16];
cx q[16],q[13];
cx q[6],q[13];
cx q[13],q[6];
cx q[3],q[6];
cx q[6],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[6],q[3];
cx q[3],q[6];
cx q[13],q[6];
cx q[6],q[13];
cx q[16],q[13];
cx q[13],q[16];
cx q[23],q[16];
cx q[16],q[23];
rz(3*pi/4) q[21];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[13];
cx q[13],q[6];
cx q[6],q[3];
rz(pi/2) q[3];
cx q[6],q[3];
cx q[13],q[6];
cx q[12],q[13];
cx q[17],q[12];
cx q[12],q[17];
cx q[8],q[7];
rz(3*pi/4) q[7];
cx q[8],q[7];
rx(pi/4) q[15];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[11],q[12];
cx q[11],q[8];
cx q[8],q[11];
cx q[8],q[1];
cx q[1],q[8];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[8];
cx q[8],q[1];
cx q[8],q[11];
cx q[11],q[8];
cx q[11],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[13],q[14];
rz(5*pi/4) q[11];
rx(pi) q[0];
rx(pi) q[1];
rx(pi) q[4];
rx(pi) q[5];
rx(pi) q[11];
rx(pi) q[24];
cx q[16],q[23];
cx q[23],q[16];
cx q[13],q[16];
cx q[16],q[13];
cx q[6],q[13];
cx q[13],q[6];
cx q[3],q[6];
cx q[6],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(pi/4) q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[2],q[3];
cx q[6],q[3];
cx q[3],q[6];
cx q[13],q[6];
cx q[6],q[13];
cx q[16],q[13];
cx q[13],q[16];
cx q[23],q[16];
cx q[16],q[23];
rz(3*pi/4) q[21];
cx q[12],q[17];
cx q[17],q[12];
cx q[12],q[13];
cx q[13],q[6];
cx q[6],q[3];
rz(pi/2) q[3];
cx q[6],q[3];
cx q[13],q[6];
cx q[12],q[13];
cx q[17],q[12];
cx q[12],q[17];
cx q[8],q[7];
rz(3*pi/4) q[7];
cx q[8],q[7];
rx(pi/4) q[15];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[11],q[12];
cx q[11],q[8];
cx q[8],q[11];
cx q[8],q[1];
cx q[1],q[8];
cx q[0],q[1];
rx(3*pi/4) q[0];
cx q[0],q[1];
cx q[1],q[8];
cx q[8],q[1];
cx q[8],q[11];
cx q[11],q[8];
cx q[11],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[13],q[14];
rz(5*pi/4) q[11];
cx q[13],q[6];
cx q[23],q[16];
cx q[16],q[23];
cx q[15],q[16];
cx q[16],q[23];
cx q[23],q[16];
cx q[6],q[13];