/**
 * Pack a Python Batch bootstrap into a `python3 -c "<stub>"` command string.
 *
 * Batch hands the JobDefinition command to ECS as part of the container override at
 * SubmitJob, and ECS caps the WHOLE override (command + merged environment +
 * resourceRequirements) at 8192 bytes. The RL and GR00T bootstraps are 14-15 KB of
 * source, so they are zlib-compressed at synth and run by a tiny decompress-and-exec
 * stub (Node deflate -> Python zlib.decompress is interoperable; exec under
 * __name__='__main__' fires the bootstrap's main()).
 *
 * Full-line `#` comments are dropped before compression. They are the bulk of what the
 * compressor cannot squeeze, they never reach the Python AST, and dropping them took the
 * RL command from 7253 to ~6000 bytes. Docstrings and inline comments are kept — removing
 * those needs a tokenizer, and the shebang line is kept as-is. test/shared.test.ts
 * proves the stripped source compiles to the same AST as the original, which is the
 * guard against a `#` line that sits inside a multi-line string literal.
 */
import * as zlib from 'zlib';

/** Drop full-line `#` comments (leading whitespace allowed); keep a first-line shebang. */
export function stripPyCommentLines(src: string): string {
  return src
    .split('\n')
    .filter((line, i) => !/^\s*#/.test(line) || (i === 0 && line.startsWith('#!')))
    .join('\n');
}

/** The `python3 -c` payload that decompresses and runs `src` as `filename`. */
export function packPyBootstrap(src: string, filename: string): string {
  const packed = zlib
    .deflateSync(Buffer.from(stripPyCommentLines(src), 'utf8'), { level: 9 })
    .toString('base64');
  return (
    'import base64,zlib;' +
    `exec(compile(zlib.decompress(base64.b64decode("${packed}")),"${filename}","exec"),` +
    '{"__name__":"__main__"})'
  );
}
