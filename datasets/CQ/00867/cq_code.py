import cadquery as cq

pnts = [(0, 10.0), (5.0, 10.0), (10.0, 12.5), (15.0, 10.0), (20.0, 15.0), (25.0, 17.5), (27.5, 15.0)]
path = cq.Workplane("XY").spline(pnts).val()
plane = cq.Plane(
path.startPoint(),
normal=path.tangentAt(0),
)
s = cq.Sketch().ellipse(1, 2, 90)
result = cq.Workplane(plane).placeSketch(s).sweep(path)
cq.exporters.export(result, 'GT.stl')