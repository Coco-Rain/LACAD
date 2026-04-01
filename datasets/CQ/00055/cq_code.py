import cadquery as cq

edge0_points = [(1, 1, 0), (1, 1, 1), (1.2, 0.74, 2)]
edge_points = [(2, 2, 0), (2, 2, 1), (2.2, 1.77, 2)]
edge0 = cq.Edge.makeSpline([cq.Vector(p) for p in edge0_points])
edge1 = cq.Edge.makeSpline([cq.Vector(p) for p in edge_points])
face = cq.Face.makeRuledSurface(edge0, edge1)
result = cq.Solid.revolve(face, 30, edge0.startPoint(), edge1.startPoint())
cq.exporters.export(result, 'GT.stl')