// Registers the resolver hook (loader.mjs) for the test run.
//
// A resolve hook runs on its own thread, so it is attached through
// module.register rather than exported from the file `--import` loads. Used as
// `node --import ./tests/register.mjs --test`.

import { register } from 'node:module'

register('./loader.mjs', import.meta.url)
