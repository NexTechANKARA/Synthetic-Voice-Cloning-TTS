// Sign-in with Supabase. Needs the global `supabase` (supabase-js) and window.VC_CONFIG (config.js).
const { supabaseUrl, supabaseKey } = window.VC_CONFIG;
export const client = window.supabase.createClient(supabaseUrl, supabaseKey);

export async function currentUser() {
  const { data } = await client.auth.getSession();
  return data.session ? data.session.user : null;
}

// On pages that need an account: send visitors who are not signed in to the sign-in page.
export async function requireUser() {
  const user = await currentUser();
  if (!user) location.replace("index.html");
  return user;
}

export async function signOut() {
  await client.auth.signOut();
  location.replace("index.html");
}

// Supabase's English error messages, rewritten to say what to do next.
export function friendlyError(error) {
  const m = (error && error.message) || "";
  if (/invalid login credentials/i.test(m)) return "That email and password don’t match. Try again, or reset your password.";
  if (/email not confirmed/i.test(m)) return "Confirm your email first. Open the link we sent you, then sign in.";
  if (/already registered|already been registered/i.test(m)) return "There’s already an account with that email. Sign in instead.";
  if (/provider is not enabled|unsupported provider/i.test(m)) return "Google sign-in isn’t switched on yet. Use your email and password for now.";
  if (/rate limit|too many/i.test(m)) return "Too many tries in a short time. Wait a minute, then try again.";
  if (/password should be at least|weak password/i.test(m)) return "Choose a longer password: at least 8 characters.";
  if (/fetch|network/i.test(m)) return "Couldn’t reach the sign-in service. Check your connection and try again.";
  return m || "Something went wrong. Try again.";
}
