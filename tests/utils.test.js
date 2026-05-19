const utils = require('../utils');

describe('utils helper functions', () => {
  test('getRsiBias returns expected labels', () => {
    expect(utils.getRsiBias(null)).toBe('sem dados');
    expect(utils.getRsiBias(75)).toBe('esticado para cima');
    expect(utils.getRsiBias(65)).toBe('forca compradora');
    expect(utils.getRsiBias(45)).toBe('zona neutra');
    expect(utils.getRsiBias(35)).toBe('pressao vendedora');
    expect(utils.getRsiBias(20)).toBe('esticado para baixo');
  });
});
