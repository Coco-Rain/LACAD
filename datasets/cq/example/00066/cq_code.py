import cadquery as cq

result = (cq.Workplane("XZ")
.hLineTo(50/2)
.spline(listOfXYTuple = [(50/2,0), (30/2,50)], tangents = [(0,1),(0,1)])
.hLineTo(0)
.close()
.revolve()
.faces(">Z or <Z")
.shell(1)
)
cq.exporters.export(result, 'GT.stl')