/**
 * Yukino DSH plugin — host half.
 *
 * This plugin deliberately does NOT inject anything into the DSH host/context.
 * All behavior lives in lib/client.js, which is discovered through the
 * package.json `dsh.client` declaration and runs only in the web profile.
 */
export function apply() {}
