import { describe, expect, it } from "vitest";
import { endpointForQueryIntent } from "./queryIntent";

describe("endpointForQueryIntent", () => {
  it("routes Ask & Answer to POST /ask", () => {
    expect(endpointForQueryIntent("ask")).toBe("/ask");
  });

  it("routes Retrieve Context to POST /context and never /ask", () => {
    expect(endpointForQueryIntent("context")).toBe("/context");
    expect(endpointForQueryIntent("context")).not.toBe("/ask");
  });
});
