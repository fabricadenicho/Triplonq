const request = require('supertest');
const app = require('../server');

describe('Server routes', () => {
  test('GET / redirects to /mnqcl.html', async () => {
    const res = await request(app).get('/');
    expect(res.statusCode).toBe(302);
    expect(res.headers.location).toBe('/mnqcl.html');
  });

  test('GET /btc returns HTML content', async () => {
    const res = await request(app).get('/btc');
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toMatch(/html/);
    expect(res.text).toMatch(/<html/i);
  });

  test('GET /cl returns HTML content', async () => {
    const res = await request(app).get('/cl');
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toMatch(/html/);
    expect(res.text).toMatch(/<html/i);
  });
});
